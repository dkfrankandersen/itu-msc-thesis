use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, s};
use std::collections::HashMap;
use crate::algs::*;
use rand::{distributions::Uniform, Rng, prelude::*};
use pq_kmeans::{PQKMeans};
use std::collections::BinaryHeap;
use crate::algs::pq_data_entry::{PQDataEntry};
use crate::algs::data_entry::{DataEntry};
use colored::*;

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
    sub_dimension: usize
}


impl ProductQuantization {
    pub fn new(verbose_print: bool, m: usize, training_size: usize, k: usize, max_iterations: usize, clusters_to_search: usize) -> Self {
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
            residuals_codebook: Array::from_elem((m, k), Array::zeros(100/m)), //TODO Dimension
            sub_dimension: 0,
        }
    }

    pub fn random_traindata(&self, dataset: ArrayView2::<f64>, train_dataset_size: usize, verbose_print: bool) -> Array2::<f64> {
        let mut rng = rand::thread_rng();
        let range = Uniform::new(0 as usize, dataset.nrows() as usize);
        let random_datapoints: Vec<usize> = (0..train_dataset_size).map(|_| rng.sample(&range)).collect();
        
        let mut train_data = Array2::zeros((train_dataset_size, dataset.ncols()));
        for (i,v) in random_datapoints.iter().enumerate() {
            let data_row = dataset.slice(s![*v,..]);
            train_data.row_mut(i).assign(&data_row);
        }
        train_data
    }

    fn kmeans(&self, k_centroids: usize, max_iterations: usize, dataset: ArrayView2::<f64>, verbose_print: bool) -> Vec::<Centroid> {
        
        let datapoint_dimension = dataset.ncols();

        // Init
        if verbose_print { println!("Centroids k-means Init"); }
        let mut centroids = Vec::<Centroid>::with_capacity(k_centroids);
        let mut rng = thread_rng();
        let dist_uniform = Uniform::new_inclusive(0, dataset.nrows()-1);
        for k in 0..k_centroids {
            let rand_key = rng.sample(dist_uniform);
            let datapoint = dataset.slice(s![rand_key,..]);
            centroids.push(Centroid{id: k, point: datapoint.to_owned(), indexes: Vec::<usize>::new()});
        }

        // Repeat
        let mut last_centroids = Vec::<Centroid>::with_capacity(k_centroids);
        if verbose_print { println!("Centroids k-means run") };
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
                    let distance = distance::cosine_similarity(&centroid.point.view() , &datapoint);
                    if best_match.0 < distance { best_match = (distance, centroid.id); }
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

    fn compute_residuals(&self, centroids: &Vec::<Centroid>, dataset: ArrayView2::<f64>, verbose_print: bool) -> Array2<f64> {
        // Compute residuals for each centroid
        if verbose_print { println!("Compute residuals for each centroid"); }
        let mut residuals = Array::from_elem((dataset.nrows(), dataset.ncols()), 0.);
        for centroid in centroids.iter() {
            for index in centroid.indexes.iter() {
                let point = dataset.slice(s![*index,..]);
                for i in 0..point.len() {
                    residuals[[*index, i]] =  point[i] - centroid.point[i];
                }
                // println!("{} {}\ndp{}\ncp: {}\nrpc: {}", centroid.id, index, point, centroid.point, residuals.slice(s![*index,..]));
            }
        }

        if verbose_print { println!("residuals, shape {:?}", residuals.shape()); }
        residuals
    }

    fn train_residuals_codebook(&self, residuals_training_data: Array2<f64>, m_subspaces: usize, k_codewords: usize, sub_dimension: usize) -> Array2::<Array1<f64>> {
        // Train residuals codebook
        let mut residuals_codebook = Array::from_elem((m_subspaces, k_codewords), Array::zeros(sub_dimension));
        for m in 0..m_subspaces {
            let begin = sub_dimension * m;
            let end = begin + sub_dimension;
            let partial_data = residuals_training_data.slice(s![.., begin..end]);
            let mut pq_kmeans = PQKMeans::new(k_codewords, 200);
            // println!("begin {}, end {}, partial_data len {}", begin, end, partial_data.len());
            // println!("LETS LOOK AT SAPCE M {} with partial_data:\n{}\n", m, partial_data);
            let codewords = pq_kmeans.run(partial_data.view());
            // println!("codewords m{} \n{:?}\n", m, codewords);
            for (k, (centroid,_)) in codewords.iter().enumerate() {
                residuals_codebook[[m,k]] = centroid.to_owned();
            }
        }
        // println!("################ residuals_codebook\n{:?}",residuals_codebook[[0,0]]);
        residuals_codebook
    }

    fn residual_encoding(&self, residuals: &Array2<f64>, residuals_codebook: &Array2::<Array1<f64>>, m_subspaces: usize, k_codewords: usize, sub_dimension: usize) -> Array1<Array1<usize>> {
        // Residuals Encoding
        let  mut pqcodes = Array::from_elem(residuals.nrows(), Array::from_elem(self.m, 0));
        for n in 0..residuals.nrows() {
            for m in 0..m_subspaces {
                let begin = sub_dimension * m;
                let end = begin + sub_dimension;
                let partial_dimension = residuals.slice(s![n, begin..end]);

                let mut best_match = (f64::NEG_INFINITY, 0);
                for k in 0..k_codewords {
                    let centroid = &residuals_codebook[[m,k]];
                    // let distance = distance::cosine_similarity(&(centroid).view(), &partial_dimension);
                    let distance = (centroid).view().dot(&partial_dimension);
                    if best_match.0 < distance { best_match = (distance, k) };
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
            let distance = distance::cosine_similarity(&centroid.point.view() , &query);
            if best_coarse_quantizers.len() < result_quantizers {
                best_coarse_quantizers.push(DataEntry {
                    index: centroid.id,  
                    distance: -distance
                });
            } else {
                let min_val = *best_coarse_quantizers.peek().unwrap();
                if distance > -min_val.distance {
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

    fn rq_pq_codes(&self, rq: Array1::<f64>, residuals_codebook: &Array2::<Array1<f64>>, m_subspaces: usize, k_codewords: usize,  sub_dimension: usize) -> Array1<usize> {
        // Compute pq codes for query residuals and get values from codebook
        let mut rq_pq_codes = Array::from_elem(m_subspaces, 0);
        for m in 0..m_subspaces {
            let begin = sub_dimension * m;
            let end = begin + sub_dimension;
            let partial_data = rq.slice(s![begin..end]);

            let mut best_match = (f64::NEG_INFINITY, 0);
            for k in 0..k_codewords {
                let centroid = &residuals_codebook[[m,k]];
                // let distance = distance::cosine_similarity(&(centroid).view(), &partial_data);
                let distance = (centroid).view().dot(&partial_data);
                if best_match.0 < distance {
                    best_match = (distance, k)
                }
            }
            rq_pq_codes[m] = best_match.1;
        }
        println!("rq_pq_codes shape {:?}", rq_pq_codes.shape());
        rq_pq_codes
    }
    
}



impl AlgorithmImpl for ProductQuantization {

    fn __str__(&self) {
        self.name.to_string();
    }

    fn fit(&mut self, dataset: ArrayView2::<f64>) {
        self.sub_dimension = dataset.ncols() / self.m;

        let verbose_print = true;
        let centroids = self.kmeans(self.m, self.max_iterations, dataset, verbose_print);
        let residuals = self.compute_residuals(&centroids, dataset, verbose_print);

        println!("{:?}", residuals);
        println!("{:?}", centroids[0].point);


        // Residuals PQ Training data
        let residuals_training_data = self.random_traindata(residuals.view(), self.training_size, true);
        if verbose_print { println!("residuals_training_data, shape {:?}", residuals_training_data.shape()); }

        self.residuals_codebook = self.train_residuals_codebook(residuals_training_data, self.m, self.k, self.sub_dimension);
        let residual_pq_codes = self.residual_encoding(&residuals, &self.residuals_codebook, self.m, self.k, self.sub_dimension);
        self.coarse_quantizer = self.compute_coarse_quantizers(&centroids, &residual_pq_codes, self.m);

    }

    fn query(&self, query: &ArrayView1::<f64>, result_count: u32) -> Vec<usize> {

        let best_coarse_quantizers = self.best_coarse_quantizers_indexes(query, &self.coarse_quantizer, self.clusters_to_search);

        println!("Best coarse_quantizers to search in {:?}", best_coarse_quantizers);

        // Lets find matches in best coarse_quantizers
        let mut best_candidates = BinaryHeap::<DataEntry>::new();
        for coarse_quantizer_index in best_coarse_quantizers.iter() {
            
            // Get coarse_quantizer from index
            let best_coares_quantizer = &self.coarse_quantizer[*coarse_quantizer_index];
            
            // Compute residuals between query and coarse_quantizer
            let rq = query.to_owned()-best_coares_quantizer.point.to_owned();

            // Compute pq codes for query residuals
            // let rq_pq_codes = &self.rq_pq_codes(rq, &self.residuals_codebook, self.m, self.k,  self.sub_dimension);
            
            // Create a distance table, for each of the M blocks to all of the K codewords -> table of size M times K.
            
            //  Get query residuals point from codebook
            // let mut rq_point = Vec::<f64>::new();
            // for (m, k) in rq_pq_codes.iter().enumerate() {
            //     let blah = &self.residuals_codebook[[m, *k]];
            //     for b in blah.iter() {
            //         rq_point.push(*b)
            //     }   
            // }
            // let arqpoint = Array::from(rq_point);

            
            for (child_key, child_values) in best_coares_quantizer.children.iter() {
                let mut point = Array::from_elem(self.m, Array::from_elem(self.sub_dimension, 0.));
                let mut c_point = Vec::<f64>::new();
                for (m, k) in child_values.iter().enumerate() {
                    // Read off the distance using the distance table
                    let blah = &self.residuals_codebook[[m, *k]];
                    point[m] = blah.to_owned();
                    for b in blah.iter() {
                        c_point.push(*b)
                    }   
                }

                let acpoint = Array::from(c_point);
                let distance = distance::cosine_similarity(&arqpoint.view() , &acpoint.view());
                // println!("acpoint {}", acpoint);

                // println!("child_key: {}, dist: {}", child_key, distance);

                if best_candidates.len() < result_count as usize {
                    best_candidates.push(DataEntry {
                        index: *child_key,  
                        distance: -distance
                    });
                } else {
                    let peek_val: DataEntry = *best_candidates.peek().unwrap();
                    if distance > -peek_val.distance {
                        let pop = best_candidates.pop();
                        best_candidates.push(DataEntry {
                            index: *child_key,  
                            distance: -distance
                        });
                    }
                }
            }
        }

               
        let mut best_n_candidates: Vec<usize> = Vec::new();
        for _ in 0..best_candidates.len() {
            let idx = (Some(best_candidates.pop()).unwrap()).unwrap();
            best_n_candidates.push(idx.index);
        }
        best_n_candidates.reverse();
        if self.verbose_print {
            println!("best_n_candidates \n{:?}", best_n_candidates);
        }
        best_n_candidates
    }

}