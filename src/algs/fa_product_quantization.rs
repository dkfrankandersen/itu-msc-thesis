use crate::util::{sampling::sampling_without_replacement};
use crate::algs::{AlgorithmImpl, AlgoParameters};
use crate::algs::distance::{DistanceMetric, min_distance};
use crate::algs::kmeans::*;
use crate::algs::common::{PQCentroid, Centroid};
use crate::util::{debug_timer::DebugTimer};
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::path::Path;
use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, s};
use rand::{prelude::*};
use ordered_float::*;
use indicatif::ProgressBar;
use bincode::serialize_into;
use std::io::{BufWriter, BufReader};

#[derive(Debug, Clone)]
pub struct FAProductQuantization {
    name: String,
    metric: String,
    algo_parameters: AlgoParameters,
    m: usize,
    training_size: usize,
    coarse_quantizer_k: usize,
    max_iterations: usize,
    verbose_print: bool,
    coarse_quantizer: Vec::<PQCentroid>,
    residuals_codebook: Array2::<Array1::<f64>>,
    residuals_codebook_k: usize,
    sub_dimension: usize,
    partial_query_begin_end: Vec::<(usize, usize)>,
    dist_metric: DistanceMetric,
}

impl FAProductQuantization {

    pub fn new(verbose_print: bool, dist_metric: DistanceMetric, algo_parameters: &AlgoParameters,
                dataset: &ArrayView2::<f64>, m: usize, coarse_quantizer_k: usize, training_size: usize, 
                residuals_codebook_k: usize, max_iterations: usize) -> Result<Self, String> {

        if m == 0 {
            return Err("m must be greater than 0".to_string());
        }
        else if dataset.ncols() % m != 0 {
            return Err(format!("M={} is not divisable with dataset dimension d={}!", m, dataset.ncols()));
        }
        else if coarse_quantizer_k == 0 {
            return Err("coarse_quantizer_k must be greater than 0".to_string());
        }
        else if training_size == 0 {
            return Err("training_size must be greater than 0".to_string());
        }
        else if training_size > dataset.nrows() {
            return Err("training_size must be less than or equal to dataset size".to_string());
        }
        else if residuals_codebook_k == 0 {
            return Err("residuals_codebook_k must be greater than 0".to_string());
        }
        else if max_iterations == 0 {
            return Err("max_iterations must be greater than 0".to_string());
        }

        Ok(FAProductQuantization {
            name: "fa_pq_TR25".to_string(),
            metric: algo_parameters.metric.clone(),
            algo_parameters: algo_parameters.clone(),
            m,         // M
            training_size,
            coarse_quantizer_k,         // K
            max_iterations,
            verbose_print,
            dist_metric,
            coarse_quantizer: Vec::<PQCentroid>::with_capacity(m),
            residuals_codebook: Array::from_elem((m, coarse_quantizer_k), Array::zeros(dataset.ncols()/m)),
            residuals_codebook_k,
            sub_dimension: dataset.ncols()/m,
            partial_query_begin_end: compute_dimension_begin_end(m, dataset.ncols()/m)
        })
    }

    pub fn random_traindata<T: RngCore>(&self, rng: T, dataset: &ArrayView2::<f64>,
                                        train_dataset_size: usize) -> Array2::<f64> {
        let unique_indexes = sampling_without_replacement(rng, dataset.nrows(), train_dataset_size);

        let mut train_data = Array2::zeros((unique_indexes.len(), dataset.ncols()));
        for (i, index) in unique_indexes.iter().enumerate() {
            let data_row = dataset.slice(s![*index,..]);
            train_data.row_mut(i).assign(&data_row);
        }
        train_data
    }

    fn compute_residual(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        let mut residual = Array1::from_elem(a.len(), 0_f64);
        for i in 0..a.len() {
            residual[i] = a[i] - b[i];
        }
        residual
    }

    fn compute_residuals(&self, centroids: &[Centroid], dataset: &ArrayView2::<f64>) -> Array2<f64> {
        // Compute residuals for each centroid
        let mut residuals = Array::from_elem((dataset.nrows(), dataset.ncols()), 0.);
        for centroid in centroids.iter() {
            for index in centroid.indexes.iter() {
                let point = dataset.slice(s![*index,..]);
                residuals.row_mut(*index).assign(&self.compute_residual(&point, &centroid.point.view()));
            }
        }
        residuals
    }

    fn train_residuals_codebook(&self, residuals_training_data: &ArrayView2<f64>, m_subspaces: usize, 
                                k_centroids: usize, sub_dimension: usize) -> Array2::<Array1<f64>> {
        // Train residuals codebook
        let mut residuals_codebook = Array::from_elem((m_subspaces, k_centroids), Array::zeros(sub_dimension));
        println!("Started Train residuals codebook with kmeans running m = {} times, with k = {} and for a max of {} iterations",
                    m_subspaces, k_centroids, self.max_iterations);
        let bar_max_iterations = ProgressBar::new((m_subspaces*self.max_iterations) as u64);

        for m in 0..m_subspaces {
            let (partial_from, partial_to) = self.partial_query_begin_end[m];
            let partial_data = residuals_training_data.slice(s![.., partial_from..partial_to]);

            let rng = thread_rng();
            let kmeans = KMeans::new(&self.dist_metric);
            let centroids = kmeans.run(rng, k_centroids, self.max_iterations, &partial_data.view(), false, &bar_max_iterations);

            for (k, centroid) in centroids.iter().enumerate() {
                residuals_codebook[[m,k]] = centroid.point.clone();
            }
        }
        bar_max_iterations.finish();
        residuals_codebook
    }

    fn residual_encoding(&self, residuals: &Array2<f64>, residuals_codebook: &Array2::<Array1<f64>>) -> Array1<Array1<usize>> {
        // Residuals Encoding
        println!("Started residual_encoding");
        let  mut pqcodes = Array::from_elem(residuals.nrows(), Array::from_elem(residuals_codebook.nrows(), 0));
        let bar_residuals = ProgressBar::new(residuals.nrows() as u64);
        for n in 0..residuals.nrows() {
            for m in 0..residuals_codebook.nrows() {
                let partial_dim = self.partial_query_begin_end[m];
                let partial_dimension = residuals.slice(s![n, partial_dim.0..partial_dim.1]);

                let mut best_distance = OrderedFloat(f64::INFINITY);
                let mut best_index = 0;
                for k in 0..residuals_codebook.ncols() {
                    let centroid = &residuals_codebook[[m,k]].view();
                    let distance = OrderedFloat(min_distance(centroid, &partial_dimension, &self.dist_metric));
                    if distance < best_distance { 
                        best_distance = distance;
                        best_index = k; 
                    };
                }
                pqcodes[n][m] = best_index;
            }
            bar_residuals.inc(1);
        }
        bar_residuals.finish();
        pqcodes
    }

    fn compute_coarse_quantizers(&self, centroids: &[Centroid], residual_pq_codes: &Array1<Array1<usize>>,
                                        m_centroids: usize) -> Vec::<PQCentroid> {
        // Compute coarse quantizer for centroids with pq codes
        let mut coarse_quantizer = Vec::<PQCentroid>::with_capacity(m_centroids);
        println!("Started compute_coarse_quantizers");
        let bar_centroids = ProgressBar::new(centroids.len() as u64);
        for centroid in centroids.iter() {
            let mut pqchilderen =  HashMap::<usize, Vec::<usize>>::new();
            for index in centroid.indexes.iter() {
                let codes = &residual_pq_codes[*index];
                pqchilderen.insert(*index, codes.to_vec());
            }
            let pqc = PQCentroid{id: centroid.id, point: centroid.point.to_owned(), children: pqchilderen};
            coarse_quantizer.push(pqc);
            bar_centroids.inc(1);
        }
        bar_centroids.finish();
        coarse_quantizer
    }
}

fn compute_dimension_begin_end(m_clusters: usize, dimension_size: usize) -> Vec::<(usize, usize)> {
    let mut result = Vec::new();
    for m in 0..m_clusters {
        let begin = dimension_size * m;
        let end = begin + dimension_size;
        result.push((begin, end));
    } 
    result
}

impl AlgorithmImpl for FAProductQuantization {

    fn name(&self) -> String {
        self.name.to_string()
    }

    fn fit(&mut self, dataset: &ArrayView2::<f64>) {
        let verbose_print = false;
        let file_residuals_codebook = &self.algo_parameters.fit_file_output("residuals_codebook");
        let file_compute_coarse_quantizers = &self.algo_parameters.fit_file_output("coarse_quantizers");

        // Load existing pre-computede data if exists
        if Path::new(file_compute_coarse_quantizers).exists() 
                                && Path::new(file_residuals_codebook).exists() {
            println!("\nFit train_residuals_codebook from file");
            let mut t = DebugTimer::start("fit train_residuals_codebook from file");
            let mut read_file = BufReader::new(File::open(file_residuals_codebook).unwrap());
            self.residuals_codebook = bincode::deserialize_from(&mut read_file).unwrap();
            t.stop();
            t.print_as_millis();
            
            println!("\nFit compute_coarse_quantizers from file");
            let mut t = DebugTimer::start("fit compute_coarse_quantizers from file");
            let mut read_file = BufReader::new(File::open(file_compute_coarse_quantizers).unwrap());
            self.coarse_quantizer = bincode::deserialize_from(&mut read_file).unwrap();
            t.stop();
            t.print_as_millis();
        } else {
            let rng = thread_rng();
            println!("\nFit run kmeans with k = {} for a max of {} iterations", self.coarse_quantizer_k, self.max_iterations);
            let mut t = DebugTimer::start("fit run kmeans");
            let bar_max_iterations = ProgressBar::new((self.max_iterations) as u64);
            let kmeans = KMeans::new(&self.dist_metric);
            let centroids = kmeans.run(rng, self.coarse_quantizer_k, self.max_iterations, dataset, verbose_print, &bar_max_iterations);
            bar_max_iterations.finish();
            t.stop();
            t.print_as_secs();

            println!("\nFit write centroids to file: {}", file_compute_coarse_quantizers);
            let mut t = DebugTimer::start("fit compute_residuals");
            let residuals = self.compute_residuals(&centroids, dataset);
            t.stop();
            t.print_as_millis();

            // Residuals PQ Training data
            let rng = thread_rng();
            let mut t = DebugTimer::start("fit random_traindata");
            let residuals_training_data = self.random_traindata(rng, &residuals.view(), self.training_size);
            t.stop();
            t.print_as_millis();

            let mut t = DebugTimer::start("fit train_residuals_codebook");
            self.residuals_codebook = self.train_residuals_codebook(&residuals_training_data.view(), self.m, 
                                                                    self.residuals_codebook_k, self.sub_dimension);
            t.stop();
            t.print_as_secs();
            
            // Write residuals_codebook to bin
            // let mut t = DebugTimer::start("Fit write residuals_codebook to file");
            // let mut new_file = BufWriter::new(File::create(file_residuals_codebook).unwrap());
            // serialize_into(&mut new_file, &self.residuals_codebook).unwrap();
            // t.stop();
            // t.print_as_millis();
 
            let mut t = DebugTimer::start("fit residual_encoding");
            let residual_pq_codes = self.residual_encoding(&residuals, &self.residuals_codebook);
            t.stop();
            t.print_as_secs();
            
            let mut t = DebugTimer::start("fit compute_coarse_quantizers");
            self.coarse_quantizer = self.compute_coarse_quantizers(&centroids, &residual_pq_codes, self.m);
            t.stop();
            t.print_as_secs();

            // Write compute_coarse_quantizers to bin
            // let mut t = DebugTimer::start("Fit write coarse_quantizer to file");
            // let mut new_file = BufWriter::new(File::create(file_compute_coarse_quantizers).unwrap());
            // serialize_into(&mut new_file, &self.coarse_quantizer).unwrap();
            // t.stop();
            // t.print_as_millis();
        }
    }
    
    fn query(&self, dataset: &ArrayView2::<f64>,  query: &ArrayView1::<f64>, results_per_query: usize,
                                                                arguments: &[usize]) -> Vec<usize> {
        
        // Query Arguments
        let clusters_to_search = arguments[0];
        let results_to_rescore = arguments[1];

        // Lets find matches in best coarse_quantizers
        // For each coarse_quantizer compute distance between query and centroid, push to heap.
        let mut best_coarse_quantizers: BinaryHeap::<(OrderedFloat::<f64>, usize)> =
                            self.coarse_quantizer.iter().map(|centroid| 
                            (OrderedFloat(-min_distance(query, &centroid.point.view(), &self.dist_metric)), centroid.id))
                            .collect();

        let min_val = std::cmp::min(clusters_to_search, best_coarse_quantizers.len());
        let best_pq_indexes: Vec::<usize> = (0..min_val).map(|_| best_coarse_quantizers
                                                                .pop().unwrap().1).collect();

        let m_dim = self.residuals_codebook.nrows();
        let k_dim = self.residuals_codebook.ncols();
        
        let mut quantizer_candidates = BinaryHeap::<(OrderedFloat::<f64>, usize)>::with_capacity(self.coarse_quantizer_k);
        let mut distance_table = Array::from_elem((m_dim, k_dim), 0.);
        
        for coarse_quantizer_index in best_pq_indexes.iter() {
            // Get coarse_quantizer from index
            let coarse_quantizer = &self.coarse_quantizer[*coarse_quantizer_index];

            let residual_qc = &self.compute_residual(&query, &coarse_quantizer.point.view());

            for m in 0..m_dim {
                let (partial_from, partial_to) = self.partial_query_begin_end[m];
                let partial_residual = residual_qc.slice(s![partial_from..partial_to]);
                for k in 0..k_dim {
                    let partial_residual_codeword = &self.residuals_codebook[[m, k]].view();
                    distance_table[[m,k]] = min_distance(&partial_residual, partial_residual_codeword, &self.dist_metric);

                }
            }

            for (child_key, child_values) in  coarse_quantizer.children.iter() {
                // Compute distance from indexes
                let mut distance: f64 = 0.;
                for (m, k) in child_values.iter().enumerate() {
                    distance += &distance_table[[m, *k]];
                }

                let distance = OrderedFloat(distance);

                // If candidates list is shorter than min results requestes push to heap
                if quantizer_candidates.len() < results_to_rescore {
                    quantizer_candidates.push((distance, *child_key));
                }
                // If distance is better, remove top (worst) and push candidate to heap
                else if distance < quantizer_candidates.peek().unwrap().0 {
                    quantizer_candidates.pop();
                    quantizer_candidates.push((distance, *child_key));
                }
            }
        }
        
        // Rescore with true distance value of query and candidates
        let candidates = &mut BinaryHeap::<(OrderedFloat::<f64>, usize)>::with_capacity(results_per_query);
        for (_, index) in quantizer_candidates.into_iter() {
            let datapoint = dataset.slice(s![index,..]);
            let distance = OrderedFloat(min_distance(query, &datapoint, &self.dist_metric));
            if candidates.len() < results_per_query {
                candidates.push((distance, index));
            } else if distance < candidates.peek().unwrap().0 {
                candidates.pop();
                candidates.push((distance, index));
            }
        }

        // Pop all candidate indexes from heap and reverse list.
        let mut best_n_candidates: Vec<usize> =  (0..candidates.len())
                                                        .map(|_| candidates
                                                        .pop().unwrap().1).collect();
        best_n_candidates.reverse();
        best_n_candidates
    }
}