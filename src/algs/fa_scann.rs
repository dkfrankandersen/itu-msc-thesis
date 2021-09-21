use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::path::Path;
use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, s};
use crate::util::{sampling::sampling_without_replacement};
use crate::algs::{AlgorithmImpl, distance::{cosine_similarity, DistanceMetric, euclidian}, AlgoParameters};
use crate::algs::{scann_kmeans::{scann_kmeans}};
use crate::algs::common::{PQCentroid, Centroid};
use crate::util::{debug_timer::DebugTimer};
use rand::{prelude::*};
pub use ordered_float::*;
use indicatif::ProgressBar;
use bincode::serialize_into;

#[derive(Debug, Clone)]
pub struct FAScann {
    name: String,
    metric: String,
    algo_parameters: AlgoParameters,
    codebook: Vec<Centroid>,
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
    anisotropic_quantization_threshold: f64
}

impl FAScann {

    pub fn new(verbose_print: bool, dist: DistanceMetric, algo_parameters: &AlgoParameters, dataset: &ArrayView2::<f64>, m: usize, coarse_quantizer_k: usize, training_size: usize, 
                            residuals_codebook_k: usize, max_iterations: usize, anisotropic_quantization_threshold: f64) -> Result<Self, String> {

        if m <= 0 {
            return Err("m must be greater than 0".to_string());
        }
        else if dataset.ncols() % m != 0 {
            return Err(format!("M={} is not divisable with dataset dimension d={}!", m, dataset.ncols()));
        }
        else if coarse_quantizer_k <= 0 {
            return Err("coarse_quantizer_k must be greater than 0".to_string());
        }
        else if training_size <= 0 {
            return Err("training_size must be greater than 0".to_string());
        }
        else if training_size > dataset.nrows() {
            return Err("training_size must be less than or equal to dataset size".to_string());
        }
        else if residuals_codebook_k <= 0 {
            return Err("residuals_codebook_k must be greater than 0".to_string());
        }
        else if max_iterations <= 0 {
            return Err("max_iterations must be greater than 0".to_string());
        }

        return Ok(FAScann {
            name: "fa_scann_c01".to_string(),
            metric: algo_parameters.metric.clone(),
            algo_parameters: algo_parameters.clone(),
            codebook: Vec::<Centroid>::new(),
            m: m,         // M
            training_size: training_size,
            coarse_quantizer_k: coarse_quantizer_k,         // K
            max_iterations: max_iterations,
            verbose_print: verbose_print,
            coarse_quantizer: Vec::<PQCentroid>::with_capacity(m),
            residuals_codebook: Array::from_elem((m, coarse_quantizer_k), Array::zeros(dataset.ncols()/m)),
            residuals_codebook_k: residuals_codebook_k,
            sub_dimension: dataset.ncols()/m,
            partial_query_begin_end: compute_dimension_begin_end(m, dataset.ncols()/m),
            anisotropic_quantization_threshold: anisotropic_quantization_threshold,
        });
    }

    pub fn random_traindata<T: RngCore>(&self, rng: T, dataset: &ArrayView2::<f64>, train_dataset_size: usize) -> Array2::<f64> {
        
        let unique_indexes = sampling_without_replacement(rng, dataset.nrows(), train_dataset_size);

        let mut train_data = Array2::zeros((unique_indexes.len(), dataset.ncols()));
        for (i, index) in unique_indexes.iter().enumerate() {
            let data_row = dataset.slice(s![*index,..]);
            train_data.row_mut(i).assign(&data_row);
        }
        train_data
    }

    fn compute_residuals(&self, centroids: &Vec::<Centroid>, dataset: &ArrayView2::<f64>) -> Array2<f64> {
        // Compute residuals for each centroid
        let mut residuals = Array::from_elem((dataset.nrows(), dataset.ncols()), 0.);
        for centroid in centroids.iter() {
            for index in centroid.indexes.iter() {
                let point = dataset.slice(s![*index,..]);
                for i in 0..point.len() {
                    residuals[[*index, i]] = point[i] - centroid.point[i];
                }
            }
        }
        residuals
    }

    fn train_residuals_codebook(&self, residuals_training_data: &ArrayView2<f64>, m_subspaces: usize, k_centroids: usize, sub_dimension: usize) -> Array2::<Array1<f64>> {
        // Train residuals codebook
        let mut residuals_codebook = Array::from_elem((m_subspaces, k_centroids), Array::zeros(sub_dimension));
        println!("Started Train residuals codebook");
        let bar_m_subspaces = ProgressBar::new(m_subspaces as u64);
        for m in 0..m_subspaces {
            let (partial_from, partial_to) = self.partial_query_begin_end[m];
            let partial_data = residuals_training_data.slice(s![.., partial_from..partial_to]);

            let rng = thread_rng();
            let centroids = scann_kmeans(rng, k_centroids, self.max_iterations, &partial_data.view(), false);

            for (k, centroid) in centroids.iter().enumerate() {
                residuals_codebook[[m,k]] = centroid.point.clone();
            }
            bar_m_subspaces.inc(1);
        }
        bar_m_subspaces.finish();
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

                let mut best_distance = OrderedFloat(f64::NEG_INFINITY);
                let mut best_index = 0;
                for k in 0..residuals_codebook.ncols() {
                    let centroid = &residuals_codebook[[m,k]].view();
                    let distance = OrderedFloat(centroid.dot(&partial_dimension));
                    // let distance = OrderedFloat(cosine_similarity(centroid,  &partial_dimension));
                    if distance > best_distance { 
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

    fn compute_coarse_quantizers(&self, centroids: &Vec::<Centroid>, residual_pq_codes: &Array1<Array1<usize>>, m_centroids: usize) -> Vec::<PQCentroid> {
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

impl AlgorithmImpl for FAScann {

    fn name(&self) -> String {
        self.name.to_string()
    }

    fn fit(&mut self, dataset: &ArrayView2::<f64>) {
        let file_fa_scann_codebook = &self.algo_parameters.fit_file_output("scann_codebook");
        
        // Load existing pre-computede data if exists
        if Path::new(file_fa_scann_codebook).exists() 
                && Path::new(file_fa_scann_codebook).exists() {
            let mut t = DebugTimer::start("fit fa_scann_codebook from file");
            let mut read_file = File::open(file_fa_scann_codebook).unwrap();
            self.codebook = bincode::deserialize_from(&mut read_file).unwrap();
            t.stop();
            t.print_as_secs();
        } else {
            // Write compute_coarse_quantizers to bin
            let rng = thread_rng();
            let mut t = DebugTimer::start("fit run kmeans");
            self.codebook = scann_kmeans(rng, self.coarse_quantizer_k, self.max_iterations, dataset, false);
            t.stop();
            t.print_as_secs();

            let mut t = DebugTimer::start("Fit write fa_scann_codebook to file");
            let mut new_file = File::create(file_fa_scann_codebook).unwrap();
            serialize_into(&mut new_file, &self.codebook).unwrap();
            t.stop();
            t.print_as_secs();
        }
    }
    
    fn query(&self, dataset: &ArrayView2::<f64>,  query: &ArrayView1::<f64>, results_per_query: usize,  arguments: &Vec::<usize>) -> Vec<usize> {
        
        // Query Arguments
        let clusters_to_search = arguments[0];
        let results_to_rescore = arguments[1];

        // Lets find matches in best coarse_quantizers
        // For each coarse_quantizer compute distance between query and centroid, push to heap.
        let mut best_coarse_quantizers: BinaryHeap::<(OrderedFloat::<f64>, usize)> = self.codebook.iter().map(|centroid| 
            (OrderedFloat(euclidian(query, &centroid.point.view())), centroid.id)
        ).collect();

        let min_val = std::cmp::min(clusters_to_search, best_coarse_quantizers.len());
        let best_coarse_quantizers_indexes: Vec::<usize> = (0..min_val).map(|_| best_coarse_quantizers.pop().unwrap().1).collect();

        println!("best_coarse_quantizers_indexes.len {}", best_coarse_quantizers_indexes.len());
        panic!("HELLO THERE!");

        let m_dim = *&self.residuals_codebook.nrows();
        let k_dim = *&self.residuals_codebook.ncols();
        
        let mut best_quantizer_candidates = BinaryHeap::<(OrderedFloat::<f64>, usize)>::with_capacity(self.coarse_quantizer_k);
        
        // Create a distance table, for each of the M blocks to all of the K codewords -> table of size M times K.
        let mut distance_table = Array::from_elem((m_dim, k_dim), 0.);
        let dim = query.len()/m_dim;
        for m in 0..m_dim {

            let begin = dim * m;
            let end = begin + dim;
            
            let partial_query = query.slice(s![begin..end]);
            for k in 0..k_dim {
                let partial_residual_codeword = &self.residuals_codebook[[m, k]].view();
                distance_table[[m,k]] = partial_residual_codeword.dot(&partial_query);
            }
        }

        for coarse_quantizer_index in best_coarse_quantizers_indexes.iter() {

            // Get coarse_quantizer from index
            let best_coares_quantizer = &self.coarse_quantizer[*coarse_quantizer_index];

            for (child_key, child_values) in  best_coares_quantizer.children.iter() {

                // Compute distance from indexes
                let mut distance: f64 = 0.;
                for (m, k) in child_values.iter().enumerate() {
                    distance += &distance_table[[m, *k]];
                }

                let neg_distance = OrderedFloat(-distance);

                // If candidates list is shorter than min results requestes push to heap
                if best_quantizer_candidates.len() < results_to_rescore {
                    best_quantizer_candidates.push((neg_distance, *child_key));
                }
                // If distance is better, remove top (worst) and push candidate to heap
                else if neg_distance < best_quantizer_candidates.peek().unwrap().0 {
                    best_quantizer_candidates.pop();
                    best_quantizer_candidates.push((neg_distance, *child_key));
                }
            }
        }
        
        // Rescore with true distance value of query and candidates
        let best_candidates = &mut BinaryHeap::<(OrderedFloat::<f64>, usize)>::with_capacity(results_per_query);
        for (_, index) in best_quantizer_candidates.into_iter() {
            let datapoint = dataset.slice(s![index,..]);
            let neg_distance = OrderedFloat(-cosine_similarity(query,  &datapoint));
            if best_candidates.len() < results_per_query {
                best_candidates.push((neg_distance, index));
            } else if neg_distance < best_candidates.peek().unwrap().0 {
                best_candidates.pop();
                best_candidates.push((neg_distance, index));
            }
        }

        // Pop all candidate indexes from heap and reverse list.
        let mut best_n_candidates: Vec<usize> =  (0..best_candidates.len()).map(|_| best_candidates.pop().unwrap().1).collect();
        best_n_candidates.reverse();
        best_n_candidates
        
    }
}