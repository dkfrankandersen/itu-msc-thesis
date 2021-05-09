use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, s};
use std::collections::{BinaryHeap, HashMap};
use rand::{distributions::Uniform, Rng, prelude::*};
pub use ordered_float::*;
use crate::algs::{AlgorithmImpl, pq_kmeans::PQKMeans, distance::cosine_similarity};
// use crate::util::{DebugTimer};

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
    coarse_quantizer_k: usize,
    max_iterations: usize,
    verbose_print: bool,
    coarse_quantizer: Vec::<PQCentroid>,
    residuals_codebook: Array2::<Array1::<f64>>,
    residuals_codebook_k: usize,
    sub_dimension: usize,
}

impl ProductQuantization {
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
        else if training_size < dataset.nrows() {
            return Err("training_size must be greater dataset size".to_string());
        }
        else if residuals_codebook_k <= 0 {
            return Err("residuals_codebook_k must be greater than 0".to_string());
        }
        else if max_iterations <= 0 {
            return Err("max_iterations must be greater than 0".to_string());
        }

        return Ok(ProductQuantization {
            name: "fa_product_quantization".to_string(),
            metric: "angular".to_string(),
            m: m,         // M
            training_size: training_size,
            coarse_quantizer_k: coarse_quantizer_k,         // K
            max_iterations: max_iterations,
            verbose_print: verbose_print,
            coarse_quantizer: Vec::<PQCentroid>::with_capacity(m),
            residuals_codebook: Array::from_elem((m, coarse_quantizer_k), Array::zeros(dataset.ncols()/m)),
            residuals_codebook_k: residuals_codebook_k,
            sub_dimension: dataset.ncols() / m,
        });
    }

    pub fn random_traindata<T: RngCore>(&self, mut rng: T, dataset: &ArrayView2::<f64>, train_dataset_size: usize) -> Array2::<f64> {
        let range = Uniform::new(0, dataset.nrows());
        let random_datapoints: Vec<usize> = (0..train_dataset_size).map(|_| rng.sample(&range)).collect();
        
        let mut train_data = Array2::zeros((random_datapoints.len(), dataset.ncols()));
        for (i, index) in random_datapoints.iter().enumerate() {
            let data_row = dataset.slice(s![*index,..]);
            train_data.row_mut(i).assign(&data_row);
        }
        train_data
    }

    fn kmeans<T: RngCore>(&self, mut rng: T, k_centroids: usize, max_iterations: usize, dataset: &ArrayView2::<f64>, verbose_print: bool) -> Vec::<Centroid> {
        
        let datapoint_dimension = dataset.ncols();

        // Init
        let mut centroids = Vec::<Centroid>::with_capacity(k_centroids);
        let dist_uniform = Uniform::new(0, dataset.nrows());
        for k in 0..k_centroids {
            let rand_index = rng.sample(dist_uniform);
            let datapoint = dataset.slice(s![rand_index,..]);
            centroids.push(Centroid{id: k, point: datapoint.to_owned(), indexes: Vec::<usize>::new()});
        }

        // Repeat
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
        let mut best_coarse_quantizers = BinaryHeap::<(OrderedFloat::<f64>, usize)>::new();
        for centroid in coarse_quantizer.iter() {
            let distance = cosine_similarity(&centroid.point.view() , &query);
            if best_coarse_quantizers.len() < result_quantizers {
                best_coarse_quantizers.push((OrderedFloat(-distance), centroid.id));
            } else {
                if OrderedFloat(distance) >= best_coarse_quantizers.peek().unwrap().0 {
                    best_coarse_quantizers.pop();
                    best_coarse_quantizers.push((OrderedFloat(-distance), centroid.id));
                }
            }
        }

        let mut result_indexes = Vec::<usize>::new();        
        for _ in 0..best_coarse_quantizers.len() {
            result_indexes.push(best_coarse_quantizers.pop().unwrap().1);
        }
        result_indexes
    }
}

fn distance_from_indexes(distance_table: &ArrayView2<f64>, child_values: &Vec::<usize>) -> f64 {
    let mut distance: f64 = 0.;
    for (m, k) in child_values.iter().enumerate() {
        distance += distance_table[[m, *k]];
    }
    distance
}

impl AlgorithmImpl for ProductQuantization {

    fn name(&self) -> String {
        self.name.to_string()
    }

    fn fit(&mut self, dataset: &ArrayView2::<f64>) {
        let verbose_print = false;
        let rng = thread_rng();
        let centroids = self.kmeans(rng, self.coarse_quantizer_k, self.max_iterations, dataset, verbose_print);
        let residuals = self.compute_residuals(&centroids, dataset);
        // Residuals PQ Training data
        let rng = thread_rng();
        let residuals_training_data = self.random_traindata(rng, &residuals.view(), self.training_size);
        self.residuals_codebook = self.train_residuals_codebook(&residuals_training_data.view(), self.m, self.residuals_codebook_k, self.sub_dimension);
        let residual_pq_codes = self.residual_encoding(&residuals, &self.residuals_codebook, self.sub_dimension);
        self.coarse_quantizer = self.compute_coarse_quantizers(&centroids, &residual_pq_codes, self.m);
    }
    
    fn query(&self, dataset: &ArrayView2::<f64>,  query: &ArrayView1::<f64>, results_per_query: usize,  arguments: &Vec::<usize>) -> Vec<usize> {

        // Query Arguments
        let clusters_to_search = arguments[0];
        let candidates_to_consider = dataset.nrows();
        let best_coarse_quantizers = self.best_coarse_quantizers_indexes(query, &self.coarse_quantizer, clusters_to_search);
        // Lets find matches in best coarse_quantizers
        let mut best_quantizer_candidates = BinaryHeap::<(OrderedFloat::<f64>, usize)>::new();
        for coarse_quantizer_index in best_coarse_quantizers.iter() {
            // Get coarse_quantizer from index
            let best_coares_quantizer = &self.coarse_quantizer[*coarse_quantizer_index];
            
            // Compute residuals between query and coarse_quantizer
            let rq = query.to_owned()-best_coares_quantizer.point.to_owned();

            // Create a distance table, for each of the M blocks to all of the K codewords -> table of size M times K.
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

            // Read off the distance using the distance table            
            for (child_key, child_values) in best_coares_quantizer.children.iter() {
                if best_quantizer_candidates.len() > candidates_to_consider {
                    break;
                }
                let distance = distance_from_indexes(&distance_table.view(), &child_values);
                best_quantizer_candidates.push((OrderedFloat(-distance),*child_key));
            }
            
            for (child_key, child_values) in best_coares_quantizer.children.iter() {
                let distance = distance_from_indexes(&distance_table.view(), &child_values);
                if OrderedFloat(distance) > -best_quantizer_candidates.peek().unwrap().0 {
                        best_quantizer_candidates.pop();
                        best_quantizer_candidates.push((OrderedFloat(-distance),*child_key));
                }
            }
        }

        // Rescore with true distance value of query and candidates
        let mut best_candidates = BinaryHeap::<(OrderedFloat::<f64>, usize)>::new();
        for candidate in best_quantizer_candidates.iter() {
            let index = candidate.1;
            let datapoint = dataset.slice(s![index,..]);
            let distance = cosine_similarity(&query.view(), &datapoint);
            
            if best_candidates.len() < results_per_query {
                best_candidates.push((OrderedFloat(-distance), index));
            } else {
                if OrderedFloat(distance) > -best_candidates.peek().unwrap().0 {
                    best_candidates.pop();
                    best_candidates.push((OrderedFloat(-distance), index));
                }
            }
        }

        let mut best_n_candidates: Vec<usize> = Vec::new();
        for _ in 0..best_candidates.len() {
            best_n_candidates.push(best_candidates.pop().unwrap().1);
        }
        best_n_candidates.reverse();
        best_n_candidates
    }

}

#[cfg(test)]
mod product_quantization_tests {
    use ndarray::{Array2, arr2};
    use assert_float_eq::*;
    use crate::algs::product_quantization::ProductQuantization;

    fn dataset1() -> Array2<f64> {
        let dataset = arr2(&[
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
            [3.0, 3.1, 3.2, 3.3, 3.4, 3.5],
            [4.0, 4.1, 4.2, 4.3, 4.4, 4.5],
            [5.0, 5.1, 5.2, 5.3, 5.4, 5.5],
            [6.0, 6.1, 6.2, 6.3, 6.4, 6.5],
            [7.0, 7.1, 7.2, 7.3, 7.4, 7.5],
            [8.0, 8.1, 8.2, 8.3, 8.4, 8.5],
            [9.0, 9.1, 9.2, 9.3, 9.4, 9.5]
        ]);

        dataset
    }

    #[test]
    fn new_d_div_m_result_ok() {
        let pq = ProductQuantization::new(false,  &dataset1().view(), 3, 1, 10, 20, 200);
        assert!(pq.is_ok());
    }
    #[test]
    fn new_d_div_m_result_err() {
        let pq = ProductQuantization::new(false,  &dataset1().view(), 5, 1, 10, 20, 200);
        assert!(pq.is_err());
    }
    #[test]
    fn new_m_par_0_result_err() {
        let pq = ProductQuantization::new(false,  &dataset1().view(), 0, 1, 10, 20, 200);
        assert!(pq.is_err());
    }
    #[test]
    fn new_clusters_par_is_0_result_err() {
        let pq = ProductQuantization::new(false,  &dataset1().view(), 3, 0, 10, 20, 200);
        assert!(pq.is_err());
    }
    #[test]
    fn new_residual_train_size_is_0_result_err() {
        let pq = ProductQuantization::new(false,  &dataset1().view(), 3, 1, 0, 20, 200);
        assert!(pq.is_err());
    }
    #[test]
    fn new_residual_train_size_is_gt_dataset_result_err() {
        let pq = ProductQuantization::new(false,  &dataset1().view(), 3, 1, 0, 20, 200);
        assert!(pq.is_err());
    }
    #[test]
    fn random_traindata_2_of_10_rows() {
        use rand::{SeedableRng, rngs::StdRng};
        let pq = ProductQuantization::new(false,  &dataset1().view(), 3, 1, 10, 20, 200);
        let rng = StdRng::seed_from_u64(11);
        let partial_dataset = pq.unwrap().random_traindata(rng, &dataset1().view(), 2);
        println!("{}", partial_dataset);
        assert!(partial_dataset.nrows() == 2);
    }
    #[test]
    fn random_traindata_6_of_6_columns() {
        use rand::{SeedableRng, rngs::StdRng};
        let pq = ProductQuantization::new(false,  &dataset1().view(), 3, 1, 10, 20, 200);
        let rng = StdRng::seed_from_u64(11);
        let partial_dataset = pq.unwrap().random_traindata(rng, &dataset1().view(), 2);
        println!("{}", partial_dataset);
        assert!(partial_dataset.ncols() == 6);
    }
    #[test]
    fn random_traindata_output_of_seed() {
        use rand::{SeedableRng, rngs::StdRng};
        let pq = ProductQuantization::new(false,  &dataset1().view(), 3, 1, 10, 20, 200);
        let rng = StdRng::seed_from_u64(11);
        let partial_dataset = pq.unwrap().random_traindata(rng, &dataset1().view(), 4);
        println!("{}", partial_dataset);
        assert!(partial_dataset == arr2(&[[2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
                                            [7.0, 7.1, 7.2, 7.3, 7.4, 7.5],
                                            [8.0, 8.1, 8.2, 8.3, 8.4, 8.5],
                                            [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]]));
    }
    #[test]
    fn kmeans_with_k_10_clusters() {
        use rand::{SeedableRng, rngs::StdRng};
        let pq = ProductQuantization::new(false,  &dataset1().view(), 3, 1, 10, 20, 200);
        let rng = StdRng::seed_from_u64(11);
        let centroids = pq.unwrap().kmeans(rng, 10, 200, &dataset1().view(), false);
        assert!(centroids.len() == 10);
    }
}