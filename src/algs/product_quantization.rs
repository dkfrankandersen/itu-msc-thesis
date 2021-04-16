use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, s};
use std::collections::HashMap;
use rand::{distributions::Uniform, Rng, prelude::*};
use std::collections::BinaryHeap;
use crate::algs::{AlgorithmImpl, pq_kmeans::PQKMeans, data_entry::DataEntry, distance::cosine_similarity};
use crate::util::{TimerDebug};
extern crate ordered_float;
pub use ordered_float::*;

#[derive(Clone, PartialEq, Debug)]
struct Centroid {
    id: usize,
    point: Array1<f64>,
    indexes: Vec::<usize>
}

#[derive(Clone, PartialEq, Debug)]
struct PQCentroid {
    id: usize,
    point: Array1<f64>,
    children: HashMap::<usize, Vec::<usize>>
}

#[derive(Debug, Clone)]
pub struct ProductQuantization {
    name: String,
    metric: String,
    m: usize,
    training_size: usize,
    k: usize,
    max_iterations: usize,
    clusters_to_search: usize,
    verbose_print: bool,
    coarse_quantizer: Vec::<PQCentroid>,
    residuals_codebook: Array2::<Array1::<f64>>,
    residuals_codebook_k: usize,
    sub_dimension: usize,
}

impl ProductQuantization {
    pub fn new(verbose_print: bool, dataset: &ArrayView2::<f64>, m: usize, k: usize, training_size: usize, 
                            residuals_codebook_k: usize, max_iterations: usize, clusters_to_search: usize) -> Self {
        ProductQuantization {
            name: "FANN_product_quantization()".to_string(),
            metric: "angular".to_string(),
            m: m,         // M
            training_size: training_size,
            k: k,         // K
            max_iterations: max_iterations,
            clusters_to_search: clusters_to_search,
            verbose_print: verbose_print,
            coarse_quantizer: Vec::<PQCentroid>::with_capacity(m),
            residuals_codebook: Array::from_elem((m, k), Array::zeros(dataset.ncols()/m)),
            residuals_codebook_k: residuals_codebook_k,
            sub_dimension: dataset.ncols() / m,
        }
    }

    pub fn random_traindata(&self, dataset: &ArrayView2::<f64>, train_dataset_size: usize) -> Array2::<f64> {
        let mut rng = rand::thread_rng();
        let range = Uniform::new(0 as usize, dataset.nrows() as usize);
        let random_datapoints: Vec<usize> = (0..train_dataset_size).map(|_| rng.sample(&range)).collect();
        
        let mut train_data = Array2::zeros((random_datapoints.len(), dataset.ncols()));
        for (i, index) in random_datapoints.iter().enumerate() {
            let data_row = dataset.slice(s![*index,..]);
            train_data.row_mut(i).assign(&data_row);
        }
        train_data
    }

    fn kmeans(&self, k_centroids: usize, max_iterations: usize, dataset: &ArrayView2::<f64>, verbose_print: bool) -> Vec::<Centroid> {
        
        let datapoint_dimension = dataset.ncols();

        // Init
        let mut centroids = Vec::<Centroid>::with_capacity(k_centroids);
        let mut rng = thread_rng();
        let dist_uniform = Uniform::new_inclusive(0, dataset.nrows()-1);
        for k in 0..k_centroids {
            let rand_index = rng.sample(dist_uniform);
            let datapoint = dataset.slice(s![rand_index,..]);
            centroids.push(Centroid{id: k, point: datapoint.to_owned(), indexes: Vec::<usize>::new()});
        }

        // Repeat
        let mut t = TimerDebug::start("Fit kmeans Repeat");
        let mut last_centroids = Vec::<Centroid>::with_capacity(k_centroids);
        for iterations in 0..max_iterations  {
            if centroids == last_centroids {
                if verbose_print { println!("Computation has converged, iterations: {}", iterations); }
                break;
            }
    
            last_centroids = centroids.clone();
    
            // Remove centroid children
            centroids.iter_mut().for_each(|c| c.indexes.clear());
            
            // Assign
            for (index, datapoint) in dataset.outer_iter().enumerate() {
                let mut best_match: (f64, usize) = (f64::NEG_INFINITY, 0);
                for centroid in centroids.iter() {
                    let distance = cosine_similarity(&centroid.point.view() , &datapoint);
                    if OrderedFloat(best_match.0) < OrderedFloat(distance) { 
                        best_match = (distance, centroid.id); 
                    }
                }
                centroids[best_match.1].indexes.push(index);
            }
            
            // Update
            for centroid in centroids.iter_mut() {
                if centroid.indexes.len() > 0 {

                    // Clear centroid point
                    centroid.point = Array::from_elem(datapoint_dimension, 0.);
                    
                    // Add dimension value of each
                    for index in centroid.indexes.iter() {
                        let point = dataset.slice(s![*index,..]);
                        for (i, x) in point.iter().enumerate() {
                            centroid.point[i] += x;
                        }
                    }

                    // Divide by indexes to get mean
                    let centroid_indexes_count = centroid.indexes.len() as f64;
                    for i in 0..datapoint_dimension {  
                        centroid.point[i] = centroid.point[i]/centroid_indexes_count;
                    }
                }
            }
        }
        t.stop_and_print_as_secs();
        centroids
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
        for m in 0..m_subspaces {
            let begin = sub_dimension * m;
            let end = begin + sub_dimension;
            let partial_data = residuals_training_data.slice(s![.., begin..end]);
            let mut pq_kmeans = PQKMeans::new(k_centroids, 200);
            let codewords = pq_kmeans.run(&partial_data.view());
            for (k, (centroid,_)) in codewords.iter().enumerate() {
                residuals_codebook[[m,k]] = centroid.to_owned();
            }
        }
        residuals_codebook
    }

    fn residual_encoding(&self, residuals: &Array2<f64>, residuals_codebook: &Array2::<Array1<f64>>, sub_dimension: usize) -> Array1<Array1<usize>> {
        // Residuals Encoding
        let  mut pqcodes = Array::from_elem(residuals.nrows(), Array::from_elem(residuals_codebook.nrows(), 0));
        for n in 0..residuals.nrows() {
            for m in 0..residuals_codebook.nrows() {
                let begin = sub_dimension * m;
                let end = begin + sub_dimension;
                let partial_dimension = residuals.slice(s![n, begin..end]);

                let mut best_match = (f64::NEG_INFINITY, 0);
                for k in 0..residuals_codebook.ncols() {
                    let centroid = &residuals_codebook[[m,k]].view();
                    let distance = centroid.dot(&partial_dimension);
                    if OrderedFloat(best_match.0) < OrderedFloat(distance) { 
                        best_match = (distance, k) 
                    };
                }
                pqcodes[n][m] = best_match.1;
            }
        }
        pqcodes
    }

    fn compute_coarse_quantizers(&self, centroids: &Vec::<Centroid>, residual_pq_codes: &Array1<Array1<usize>>, m_centroids: usize) -> Vec::<PQCentroid> {
        // Compute coarse quantizer for centroids with pq codes
        let mut coarse_quantizer = Vec::<PQCentroid>::with_capacity(m_centroids);
        for centroid in centroids.iter() {
            let mut pqchilderen =  HashMap::<usize, Vec::<usize>>::new();
            for index in centroid.indexes.iter() {
                let codes = &residual_pq_codes[*index];
                pqchilderen.insert(*index, codes.to_vec());
            }
            let pqc = PQCentroid{id: centroid.id, point: centroid.point.to_owned(), children: pqchilderen};
            coarse_quantizer.push(pqc);
        }
        coarse_quantizer
    }

    fn best_coarse_quantizers_indexes(&self, query: &ArrayView1::<f64>, coarse_quantizer: &Vec::<PQCentroid>, result_quantizers: usize) -> Vec::<usize> {
        // Find best coarse_quantizer
        let mut best_coarse_quantizers = BinaryHeap::<DataEntry>::new();
        for centroid in coarse_quantizer.iter() {
            let distance = cosine_similarity(&centroid.point.view() , &query);
            if best_coarse_quantizers.len() < result_quantizers {
                best_coarse_quantizers.push(DataEntry {
                    index: centroid.id,  
                    distance: -distance
                });
            } else {
                let min_val = best_coarse_quantizers.peek().unwrap();
                if OrderedFloat(distance) > OrderedFloat(-min_val.distance) {
                    best_coarse_quantizers.pop();
                    best_coarse_quantizers.push(DataEntry {
                        index: centroid.id,  
                        distance: -distance
                    });
                }
            }
        }

        let mut result_indexes = Vec::<usize>::new();        
        for _ in 0..best_coarse_quantizers.len() {
            let data_entry = best_coarse_quantizers.pop().unwrap();
            result_indexes.push(data_entry.index);
        }
        result_indexes
    }
}

impl AlgorithmImpl for ProductQuantization {

    fn __str__(&self) {
        self.name.to_string();
    }

    fn fit(&mut self, dataset: &ArrayView2::<f64>) {
        self.sub_dimension = dataset.ncols() / self.m;

        let verbose_print = false;
        let mut t = TimerDebug::start("Fit kmeans");
        let centroids = self.kmeans(self.m, self.max_iterations, dataset, verbose_print);
        t.stop_and_print_as_secs();
        let mut t = TimerDebug::start("Fit compute_residuals");
        let residuals = self.compute_residuals(&centroids, dataset);
        t.stop_and_print_as_millis();

        // Residuals PQ Training data
        
        let mut t = TimerDebug::start("Fit random_traindata");
        let residuals_training_data = self.random_traindata(&residuals.view(), self.training_size);
        t.stop_and_print_as_millis();
        let mut t = TimerDebug::start("Fit train_residuals_codebook");
        self.residuals_codebook = self.train_residuals_codebook(&residuals_training_data.view(), self.m, self.residuals_codebook_k, self.sub_dimension);
        t.stop_and_print_as_secs();
        let mut t = TimerDebug::start("Fit residual_encoding");
        let residual_pq_codes = self.residual_encoding(&residuals, &self.residuals_codebook, self.sub_dimension);
        t.stop_and_print_as_secs();
        let mut t = TimerDebug::start("Fit compute_coarse_quantizers");
        self.coarse_quantizer = self.compute_coarse_quantizers(&centroids, &residual_pq_codes, self.m);
        t.stop_and_print_as_millis();
    }
    
    fn query(&self, dataset: &ArrayView2::<f64>,  query: &ArrayView1::<f64>, result_count: usize) -> Vec<usize> {
        let mut t = TimerDebug::start("Query best_coarse_quantizers");
        let best_coarse_quantizers = self.best_coarse_quantizers_indexes(query, &self.coarse_quantizer, self.clusters_to_search);
        t.stop_and_print_as_millis();
        // Lets find matches in best coarse_quantizers
        // let mut best_quantizer_candidates = BinaryHeap::<DataEntry>::new();
        let mut best_quantizer_candidates = BinaryHeap::<(OrderedFloat::<f64>, usize)>::new();
        let candidates_to_consider = result_count * 1000;
        for coarse_quantizer_index in best_coarse_quantizers.iter() {
            // Get coarse_quantizer from index
            let best_coares_quantizer = &self.coarse_quantizer[*coarse_quantizer_index];
            
            // Compute residuals between query and coarse_quantizer
            let rq = query.to_owned()-best_coares_quantizer.point.to_owned();

            // Create a distance table, for each of the M blocks to all of the K codewords -> table of size M times K.
            let mut t = TimerDebug::start("Query Create a distance table");
            let mut distance_table = Array::from_elem((self.m, self.residuals_codebook_k), 0.);
            for m in 0..self.m {
                let begin = self.sub_dimension * m;
                let end = begin + self.sub_dimension;
                let partial_query = rq.slice(s![begin..end]);
                for k in 0..self.residuals_codebook_k {
                    let partial_residual_codeword = &self.residuals_codebook[[m, k]].view();
                    distance_table[[m,k]] = partial_residual_codeword.dot(&partial_query);
                }
            }
            t.stop_and_print_as_millis();
            // Read off the distance using the distance table            
            let mut t = TimerDebug::start("Query Read off the distance using the distance table");
            for (child_key, child_values) in best_coares_quantizer.children.iter() {
                if best_quantizer_candidates.len() > candidates_to_consider {
                    break;
                }
                let mut distance: f64 = 0.;
                for (m, k) in child_values.iter().enumerate() {
                    let m_dist = distance_table[[m,*k]];
                    distance += m_dist;
                }
                best_quantizer_candidates.push((OrderedFloat(-distance),*child_key));
            }
            
            
            for (child_key, child_values) in best_coares_quantizer.children.iter() {
                let mut distance: f64 = 0.;
                for (m, k) in child_values.iter().enumerate() {
                    let m_dist = distance_table[[m,*k]];
                    distance += m_dist;
                }
                if OrderedFloat(distance) > -best_quantizer_candidates.peek().unwrap().0 {
                        best_quantizer_candidates.pop();
                        best_quantizer_candidates.push((OrderedFloat(-distance),*child_key));
                }
            }
            t.stop_and_print_as_millis();
        }

        let mut t = TimerDebug::start("Query Rescore with true distance value");
        // Rescore with true distance value of query and candidates
        let mut best_candidates = BinaryHeap::<DataEntry>::new();
        for data_entry in best_quantizer_candidates.iter() {
            let index = data_entry.1;
            let datapoint = dataset.slice(s![index,..]);
            let distance = cosine_similarity(&query.view(), &datapoint);
            
            if best_candidates.len() < result_count {
                best_candidates.push(DataEntry {
                    index: index,  
                    distance: -distance
                });
            } else {
                let peek_val: DataEntry = *best_candidates.peek().unwrap();
                if OrderedFloat(distance) > OrderedFloat(-peek_val.distance) {
                    best_candidates.pop();
                    best_candidates.push(DataEntry {
                        index: index,  
                        distance: -distance
                    });
                }
            }
        }
        t.stop_and_print_as_millis();
               
        let mut best_n_candidates: Vec<usize> = Vec::new();
        for _ in 0..best_candidates.len() {
            let idx = (Some(best_candidates.pop()).unwrap()).unwrap();
            best_n_candidates.push(idx.index);
        }
        best_n_candidates.reverse();
        
        best_n_candidates
    }

}