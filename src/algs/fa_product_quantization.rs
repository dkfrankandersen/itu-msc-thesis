use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::path::Path;
use std::thread;
use std::sync::{Mutex, Arc};
use std::ptr::NonNull;
use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, s};
use crate::util::{sampling::sampling_without_replacement};
use crate::algs::{AlgorithmImpl, distance::cosine_similarity};
use crate::algs::{kmeans::{kmeans}};
use crate::algs::common::{PQCentroid, Centroid};
use crate::util::{DebugTimer};
use rand::{prelude::*};
pub use ordered_float::*;
use indicatif::ProgressBar;
use bincode::serialize_into;
use rayon::prelude::*;


#[derive(Debug, Clone)]
pub struct FAProductQuantization {
    name: String,
    metric: String,
    m: usize,
    training_size: usize,
    coarse_quantizer_k: usize,
    max_iterations: usize,
    verbose_print: bool,
    coarse_quantizer: Vec::<PQCentroid>,
    residuals_codebook: Array2::<Array1::<f64>>,
    residuals_codebook_k: usize,
    sub_dimension: usize,
    // partial_query_begin_end: HashMap::<usize, (usize, usize)>
    partial_query_begin_end: Vec::<(usize, usize)>

}

unsafe impl Send for FAProductQuantization {}
unsafe impl Sync for FAProductQuantization {}

impl FAProductQuantization {

    pub fn new(verbose_print: bool, dataset: &ArrayView2::<f64>, m: usize, coarse_quantizer_k: usize, training_size: usize, 
                            residuals_codebook_k: usize, max_iterations: usize) -> Result<Self, String> {

        if m <= 0 {
            return Err("m must be greater than 0".to_string());
        }
        else if dataset.ncols() % m != 0 {
            return Err("M is not divisable with dataset dimension!".to_string());
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

        return Ok(FAProductQuantization {
            name: "fa_pq_REF_0605_1545_ref".to_string(),
            metric: "angular".to_string(),
            m: m,         // M
            training_size: training_size,
            coarse_quantizer_k: coarse_quantizer_k,         // K
            max_iterations: max_iterations,
            verbose_print: verbose_print,
            coarse_quantizer: Vec::<PQCentroid>::with_capacity(m),
            residuals_codebook: Array::from_elem((m, coarse_quantizer_k), Array::zeros(dataset.ncols()/m)),
            residuals_codebook_k: residuals_codebook_k,
            sub_dimension: dataset.ncols()/m,
            partial_query_begin_end: compute_dimension_begin_end(m, dataset.ncols()/m)
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
            let centroids = kmeans(rng, k_centroids, self.max_iterations, &partial_data.view(), false);

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
                    if best_distance < distance { 
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

    fn best_coarse_quantizers_indexes(&self, query: &ArrayView1::<f64>, coarse_quantizer: &Vec::<PQCentroid>, clusters_to_search: usize) -> Vec::<usize> {

        // For each coarse_quantizer compute distance between query and centroid, push to heap.
        let mut best_coarse_quantizers: BinaryHeap::<(OrderedFloat::<f64>, usize)> = coarse_quantizer.par_iter().map(|centroid| 
            (OrderedFloat(cosine_similarity(query, &centroid.point.view())), centroid.id)
        ).collect();

        let min_val = std::cmp::min(clusters_to_search, best_coarse_quantizers.len());
        let result_indexes: Vec::<usize> = (0..min_val).map(|_| best_coarse_quantizers.pop().unwrap().1).collect();
        result_indexes
    }

    fn query_type1(&self, dataset: &ArrayView2::<f64>,  query: &ArrayView1::<f64>, results_per_query: usize,  arguments: &Vec::<usize>) -> Vec<usize> {
        // Query Arguments
        let clusters_to_search = arguments[0];
        let heap_size = arguments[1];
        
        // Lets find matches in best coarse_quantizers
        let best_coarse_quantizers_indexes = self.best_coarse_quantizers_indexes(query, &self.coarse_quantizer, clusters_to_search);

        let mut best_quantizer_candidates = BinaryHeap::<(OrderedFloat::<f64>, usize)>::with_capacity(self.coarse_quantizer_k);
        for coarse_quantizer_index in best_coarse_quantizers_indexes.iter() {
            // Get coarse_quantizer from index
            let best_coares_quantizer = &self.coarse_quantizer[*coarse_quantizer_index];
            
            // Compute residuals between query and coarse_quantizer
            let residual_point = best_coares_quantizer.compute_residual(query.to_owned());

            // Create a distance table, for each of the M blocks to all of the K codewords -> table of size M times K.
            let distance_table = best_coares_quantizer.compute_distance_table(&residual_point, &self.residuals_codebook);
            let dist_and_keys = best_coares_quantizer.approximated_distances_with_keys(&distance_table);
            let quantizer_candidates = &mut best_coares_quantizer.approximated_candidates(dist_and_keys, heap_size);
            best_quantizer_candidates.append(quantizer_candidates);
        }

        // Remove worst candidates
        while best_quantizer_candidates.len() > heap_size {
                best_quantizer_candidates.pop();
        }

        // Rescore with true distance value of query and candidates
        let best_candidates = &mut BinaryHeap::<(OrderedFloat::<f64>, usize)>::with_capacity(results_per_query);
        for candidate in best_quantizer_candidates.iter() {
            let index = candidate.1;
            let datapoint = dataset.slice(s![index,..]);
            let neg_distance = OrderedFloat(-cosine_similarity(query,  &datapoint));
            if best_candidates.len() < results_per_query {
                best_candidates.push((neg_distance, index));
            } else if neg_distance < best_candidates.peek().unwrap().0 {
                best_candidates.pop();
                best_candidates.push((neg_distance, index));
            }
        }

        // Remove worst elements from heap, and extract best index worst to best.
        let mut best_n_candidates: Vec<usize> = Vec::with_capacity(results_per_query);
        for _ in 0..best_candidates.len() {
            let c = best_candidates.pop();
            best_n_candidates.push(c.unwrap().1);
        }
        
        // Invert best indexes to get best to worst
        best_n_candidates.reverse();
        best_n_candidates
    }

    fn query_type2(&self, dataset: &ArrayView2::<f64>,  query: &ArrayView1::<f64>, results_per_query: usize,  arguments: &Vec::<usize>) -> Vec<usize> {
        // Query Arguments
        let clusters_to_search = arguments[0];
        let heap_size = arguments[1];
        
        let best_coarse_quantizers_indexes = self.best_coarse_quantizers_indexes(query, &self.coarse_quantizer, clusters_to_search);

        // Lets find matches in best coarse_quantizers
        let collect_best_quantizers: Vec::<PQCentroid> = best_coarse_quantizers_indexes.into_par_iter().map(|index| {
            let best_coares_quantizer = self.coarse_quantizer[index].clone();
            best_coares_quantizer
        }).collect();

        let candidates_per_quantizers: Vec::<_> = collect_best_quantizers.par_iter().map(|quantizer| {
            let residual_point = quantizer.compute_residual(query.to_owned());
            let distance_table = quantizer.compute_distance_table(&residual_point, &self.residuals_codebook);
            let dist_and_keys = quantizer.approximated_distances_with_keys(&distance_table);
            let candidates = quantizer.approximated_candidates(dist_and_keys, heap_size);
            candidates
        }).collect();

        let mut best_quantizer_candidates: BinaryHeap::<_> = candidates_per_quantizers.into_par_iter().flatten().collect();

        // Remove worst candidates
        while best_quantizer_candidates.len() > heap_size {
                best_quantizer_candidates.pop();
        }

        // Rescore with true distance value of query and candidates
        let best_candidates = &mut BinaryHeap::<(OrderedFloat::<f64>, usize)>::with_capacity(results_per_query);
        for candidate in best_quantizer_candidates.iter() {
            let index = candidate.1;
            let datapoint = dataset.slice(s![index,..]);
            let neg_distance = OrderedFloat(-cosine_similarity(query,  &datapoint));
            if best_candidates.len() < results_per_query {
                best_candidates.push((neg_distance, index));
            } else if neg_distance < best_candidates.peek().unwrap().0 {
                best_candidates.pop();
                best_candidates.push((neg_distance, index));
            }
        }

        // Remove worst elements from heap, and extract best index worst to best.
        let mut best_n_candidates: Vec<usize> = Vec::with_capacity(results_per_query);
        for _ in 0..best_candidates.len() {
            let c = best_candidates.pop();
            best_n_candidates.push(c.unwrap().1);
        }
        
        // Invert best indexes to get best to worst
        best_n_candidates.reverse();
        best_n_candidates
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
        let fit_from_file = true;
        let file_residuals_codebook = "saved_objects/residuals_codebook.bin";
        let file_compute_coarse_quantizers = "saved_objects/compute_coarse_quantizers.bin";

        // Load existing pre-computede data if exists
        if fit_from_file && Path::new(file_compute_coarse_quantizers).exists() 
                                && Path::new(file_residuals_codebook).exists() {
            let mut t = DebugTimer::start("fit train_residuals_codebook from file");
            let mut read_file = File::open(file_residuals_codebook).unwrap();
            self.residuals_codebook = bincode::deserialize_from(&mut read_file).unwrap();
            t.stop();
            t.print_as_secs();

            let mut t = DebugTimer::start("fit compute_coarse_quantizers from file");
            let mut read_file = File::open(file_compute_coarse_quantizers).unwrap();
            self.coarse_quantizer = bincode::deserialize_from(&mut read_file).unwrap();
            t.stop();
            t.print_as_secs();
        } else {
            let rng = thread_rng();
            let mut t = DebugTimer::start("fit run kmeans");
            let centroids = kmeans(rng, self.coarse_quantizer_k, self.max_iterations, dataset, verbose_print);
            t.stop();
            t.print_as_secs();

            let mut t = DebugTimer::start("fit compute_residuals");
            let residuals = self.compute_residuals(&centroids, dataset);
            t.stop();
            t.print_as_secs();

            // Residuals PQ Training data
            let rng = thread_rng();
            let mut t = DebugTimer::start("fit random_traindata");
            let residuals_training_data = self.random_traindata(rng, &residuals.view(), self.training_size);
            t.stop();
            t.print_as_secs();
            

            let mut t = DebugTimer::start("fit train_residuals_codebook");
            self.residuals_codebook = self.train_residuals_codebook(&residuals_training_data.view(), self.m, self.residuals_codebook_k, self.sub_dimension);
            t.stop();
            t.print_as_secs();
            
            // Write residuals_codebook to bin
            let mut t = DebugTimer::start("Fit write residuals_codebook to file");
            let mut new_file = File::create(file_residuals_codebook).unwrap();
            serialize_into(&mut new_file, &self.residuals_codebook).unwrap();
            t.stop();
            t.print_as_secs();
 
            let mut t = DebugTimer::start("fit residual_encoding");
            let residual_pq_codes = self.residual_encoding(&residuals, &self.residuals_codebook);
            t.stop();
            t.print_as_secs();
            
            let mut t = DebugTimer::start("fit compute_coarse_quantizers");
            self.coarse_quantizer = self.compute_coarse_quantizers(&centroids, &residual_pq_codes, self.m);
            t.stop();
            t.print_as_secs();

            // Write compute_coarse_quantizers to bin
            let mut t = DebugTimer::start("Fit write compute_coarse_quantizers to file");
            let mut new_file = File::create(file_compute_coarse_quantizers).unwrap();
            serialize_into(&mut new_file, &self.coarse_quantizer).unwrap();
            t.stop();
            t.print_as_secs();
        }
    }
    
    fn query(&self, dataset: &ArrayView2::<f64>,  query: &ArrayView1::<f64>, results_per_query: usize,  arguments: &Vec::<usize>) -> Vec<usize> {
        
        let mut t1 = DebugTimer::start("query_type1");
        let query_type1 = self.query_type1(dataset, query, results_per_query, arguments);
        t1.stop();
        let mut t2 = DebugTimer::start("query_type2");
        let query_type2 = self.query_type2(dataset, query, results_per_query, arguments);
        t2.stop();
        t1.print_as_nanos();
        t2.print_as_nanos();
        println!("Equal {}", query_type1==query_type2);
        query_type1
    }
}

#[cfg(test)]
mod product_quantization_tests {
    use ndarray::{Array2, arr2};
    use crate::algs::fa_product_quantization::FAProductQuantization;

    fn dataset1() -> Array2<f64> {
        arr2(&[
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
            [3.0, 3.1, 3.2, 3.3, 3.4, 3.5],
            [4.0, 4.1, 4.2, 4.3, 4.4, 4.5],
            [5.0, 5.1, 5.2, 5.3, 5.4, 5.5],
            [6.0, 6.1, 6.2, 6.3, 6.4, 6.5],
            [7.0, 7.1, 7.2, 7.3, 7.4, 7.5],
            [8.0, 8.1, 8.2, 8.3, 8.4, 8.5],
            [9.0, 9.1, 9.2, 9.3, 9.4, 9.5],
        ])
    }

    #[test]
    fn new_d_div_m_result_ok() {
        let pq = FAProductQuantization::new(false,  &dataset1().view(), 3, 1, 10, 20, 200);
        assert!(pq.is_ok());
    }
    #[test]
    fn new_d_div_m_result_err() {
        let pq = FAProductQuantization::new(false,  &dataset1().view(), 5, 1, 10, 20, 200);
        assert!(pq.is_err());
    }
    #[test]
    fn new_m_par_0_result_err() {
        let pq = FAProductQuantization::new(false,  &dataset1().view(), 0, 1, 10, 20, 200);
        assert!(pq.is_err());
    }
    #[test]
    fn new_clusters_par_is_0_result_err() {
        let pq = FAProductQuantization::new(false,  &dataset1().view(), 3, 0, 10, 20, 200);
        assert!(pq.is_err());
    }
    #[test]
    fn new_residual_train_size_is_0_result_err() {
        let pq = FAProductQuantization::new(false,  &dataset1().view(), 3, 1, 0, 20, 200);
        assert!(pq.is_err());
    }
    #[test]
    fn new_residual_train_size_is_gt_dataset_result_err() {
        let pq = FAProductQuantization::new(false,  &dataset1().view(), 3, 1, 0, 20, 200);
        assert!(pq.is_err());
    }
    #[test]
    fn random_traindata_2_of_10_rows() {
        use rand::{SeedableRng, rngs::StdRng};
        let pq = FAProductQuantization::new(false,  &dataset1().view(), 3, 1, 10, 20, 200);
        let rng = StdRng::seed_from_u64(11);
        let partial_dataset = pq.unwrap().random_traindata(rng, &dataset1().view(), 2);
        println!("{}", partial_dataset);
        assert!(partial_dataset.nrows() == 2);
    }
    #[test]
    fn random_traindata_6_of_6_columns() {
        use rand::{SeedableRng, rngs::StdRng};
        let pq = FAProductQuantization::new(false,  &dataset1().view(), 3, 1, 10, 20, 200);
        let rng = StdRng::seed_from_u64(11);
        let partial_dataset = pq.unwrap().random_traindata(rng, &dataset1().view(), 2);
        assert!(partial_dataset.ncols() == 6);
    }
    #[test]
    fn random_traindata_output_of_seed() {
        use rand::{SeedableRng, rngs::StdRng};
        let pq = FAProductQuantization::new(false,  &dataset1().view(), 3, 1, 10, 20, 200);
        let rng = StdRng::seed_from_u64(11);
        let partial_dataset = pq.unwrap().random_traindata(rng, &dataset1().view(), 4);
        println!("{}",partial_dataset);
        assert!(partial_dataset == arr2(&[[5.0, 5.1, 5.2, 5.3, 5.4, 5.5],
                                            [8.0, 8.1, 8.2, 8.3, 8.4, 8.5],
                                            [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
                                            [4.0, 4.1, 4.2, 4.3, 4.4, 4.5]]));
    }
}