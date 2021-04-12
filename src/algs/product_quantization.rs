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
struct PQCentroid {
    id: usize,
    point: Array1<f64>,
    children: HashMap::<usize, Vec::<usize>>
}

#[derive(Debug, Clone)]
pub struct ProductQuantization {
    name: String,
    metric: String,
    // dataset: Option<Array2::<f64>>,
    // codebook: Option::<Array2::<Array1::<f64>>>,
    // pqcodes: Option::<Array2::<usize>>,
    m: usize,
    training_size: usize,
    k: usize,
    max_iterations: usize,
    clusters_to_search: usize,
    verbose_print: bool,
    dimension: usize,
    sub_dimension: usize,
    coarse_quantizer: Vec::<PQCentroid>,
    residuals_codebook: Array2::<Array1::<f64>>,
}


impl ProductQuantization {
    pub fn new(verbose_print: bool, m: usize, training_size: usize, k: usize, max_iterations: usize, clusters_to_search: usize) -> Self {
        ProductQuantization {
            name: "FANN_product_quantization()".to_string(),
            metric: "angular".to_string(),
            // dataset: None,
            // codebook: None,
            // pqcodes: None,
            m: m,         // M
            training_size: training_size,
            k: k,         // K
            max_iterations: max_iterations,
            clusters_to_search: clusters_to_search,
            verbose_print: verbose_print,
            dimension: 0,
            sub_dimension: 0,
            coarse_quantizer: Vec::<PQCentroid>::with_capacity(m),
            residuals_codebook: Array::from_elem((m, k), Array::zeros(100/m)) //TODO Dimension
        }
    }

    pub fn random_traindata(&self, dataset: ArrayView2::<f64>, train_dataset_size: usize) -> Array2::<f64> {
        let mut rng = rand::thread_rng();
        let range = Uniform::new(0 as usize, dataset.nrows() as usize);
        let random_datapoints: Vec<usize> = (0..train_dataset_size).map(|_| rng.sample(&range)).collect();
        println!("Random datapoints [{}] for training, between [0..{}]", train_dataset_size, dataset.nrows());
        
        let mut train_data = Array2::zeros((train_dataset_size, dataset.ncols()));
        for (i,v) in random_datapoints.iter().enumerate() {
            let data_row = dataset.slice(s![*v,..]);
            train_data.row_mut(i).assign(&data_row);
        }
        train_data
    }

    // fn train_codebook(&mut self, train_data: ArrayView2::<f64>) {
    //     // Create codebook, [m,k,[sd]] m-th subspace, k-th codewords, sd-th dimension
    //     let mut codebook = Array::from_elem((self.m, self.k), Array::zeros(self.sub_dimension));
    //     for m in 0..self.m {
    //         let begin = self.sub_dimension * m;
    //         let end = begin + self.sub_dimension - 1;
    //         let partial_data = train_data.slice(s![.., begin..end]);
    //         if self.verbose_print {
    //             // println!("Run k-means for m [{}], sub dim {:?}, first element [{}]", m, partial_data.shape(), partial_data[[0,0]]);
    //         }
    //         let mut pq_kmeans = PQKMeans::new(self.k, self.max_iterations);
    //         let codewords = pq_kmeans.run(partial_data.view());
    //         for (k, (centroid,_)) in codewords.iter().enumerate() {
    //                 codebook[[m,k]] = centroid.to_owned();
    //         }
    //     }
    //     self.codebook = Some(codebook);
    // }

    // fn dataset_to_pqcodes(&self, dataset: ArrayView2::<f64>) -> Array2::<usize> {
    //     let mut pqcodes = Array::from_elem((dataset.nrows(), self.m), 0);
    //     for n in 0..dataset.nrows() {
    //         for m in 0..self.m {
    //             let begin = self.sub_dimension * m;
    //             let end = begin + self.sub_dimension - 1;
    //             let partial_data = dataset.slice(s![n, begin..end]);

    //             let mut best_centroid = 0;
    //             let mut best_distance = f64::NEG_INFINITY;

    //             for k in 0..self.k {
    //                 let centroid = &self.codebook.as_ref().unwrap()[[m,k]];
    //                 let distance = distance::cosine_similarity(&(centroid).view(), &partial_data);
    //                 if best_distance < distance {
    //                     best_centroid = k;
    //                     best_distance = distance;
    //                 }
    //             }
    //                 pqcodes[[n, m]] = best_centroid;
    //         }   
    //     }
    //     pqcodes
    // }

    // fn distance_table(&self, query: &ArrayView1<f64>) -> Vec::<Vec::<f64>> {
    //     let mut dtable = Vec::<Vec::<f64>>::with_capacity(self.m);
        
    //     for m in 0..self.m {
    //         dtable.push(Vec::with_capacity(self.k));
    //         let begin = self.sub_dimension * m;
    //         let end = begin + self.sub_dimension - 1;
    //         let partial_data = &query.slice(s![begin..end]);
    //         for k in 0..self.k {
    //             let code = &self.codebook.as_ref().unwrap();
    //             let sub_centroid = &code[[m,k]];
                
    //             // let dist = distance::cosine_similarity(&sub_centroid.view(), &partial_data);
    //             let dist = (sub_centroid.view()).dot(partial_data);
    //             dtable[m].push(dist);
    //         }
    //     }

    //     dtable
    // }

    // fn adist(&self, dtable: Vec::<Vec::<f64>>) -> Array1::<f64> {
    //     let n = self.dataset.as_ref().unwrap().nrows();
    //     let pqcode = self.pqcodes.as_ref().unwrap();
    //     let mut dists = Array1::<f64>::zeros(n);
    //     for n in 0..n {
    //         for m in 0..self.m {
    //             if n == 97478 {
    //                 println!("%%%%%%%%%%%%%%% \nadist for {} m: {} pqcode: {} dist {}", n, m, pqcode[[n,m]], dtable[m][pqcode[[n,m]]]);
    //             }
    //             dists[n] += dtable[m][pqcode[[n,m]]];
    //         }
    //         if n == 97478 {
    //             println!("%%%%%%%%%%%%%%% \nadist for dists[{}] {}", n, dists[n]);
    //         }
    //     }
    //     dists
    // }

    fn invertedList(&mut self, dataset: ArrayView2::<f64>) {
        
        let max_iterations = 200;

        #[derive(Clone, PartialEq, Debug)]
        struct Centroid {
            id: usize,
            point: Array1<f64>,
            children: Vec::<usize>
        }

        let datapoint_dimension = dataset.ncols();

        // Init
        println!("Centroids k-means Init");
        let mut centroids = Vec::<Centroid>::with_capacity(self.m);
        let mut rng = thread_rng();
        let dist_uniform = Uniform::new_inclusive(0, dataset.nrows()-1);
        for i in 0..self.m {
            let rand_key = rng.sample(dist_uniform);
            let datapoint = dataset.slice(s![rand_key,..]);
            centroids.push(Centroid{id: i, point: datapoint.to_owned(), children: Vec::<usize>::new()});
        }

        // Repeat
        let mut last_centroids = Vec::<Centroid>::with_capacity(self.m);
        let mut iterations = 1;
        println!("Centroids k-means run");
        loop  {
            if iterations > max_iterations {
                if self.verbose_print {
                    println!("Max iterations reached, iterations: {}", iterations-1);
                }
                break;
            } else if centroids == last_centroids {
                if self.verbose_print {
                    println!("Computation has converged, iterations: {}", iterations-1);
                }
                break;
            }
    
            last_centroids = centroids.clone();
    
            // Remove centroid children
            centroids.iter_mut().for_each(|c| c.children.clear());
            
            // Assign
            for (idx, candidate) in dataset.outer_iter().enumerate() {
                let mut best_match: (f64, usize) = (f64::NEG_INFINITY, 0);
                for (k, centroid) in centroids.iter().enumerate() {
                    let distance = distance::cosine_similarity(&centroid.point.view() , &candidate);
                    if best_match.0 < distance { best_match = (distance, k); }
                }
                centroids[best_match.1].children.push(idx);
            }
            
            // Update
            for centroid in centroids.iter_mut() {
                if centroid.children.len() > 0 {
                    centroid.point = Array::from_elem(datapoint_dimension, 0.);
                    
                    for child_key in centroid.children.iter() {
                        let child_point = dataset.slice(s![*child_key,..]);
                        for (i, x) in child_point.iter().enumerate() {
                            centroid.point[i] += x;
                        }
                    }

                    for i in 0..datapoint_dimension {  
                        centroid.point[i] = centroid.point[i]/centroid.children.len() as f64;
                    }
                }
            }
            iterations += 1;
        }
        
        // Compute residuals for each centroid
        println!("Compute residuals for each centroid");
        // let mut residuals = Array::from_elem((dataset.nrows(), datapoint_dimension), 0.);
        let mut residuals = Array::from_elem((dataset.nrows(), dataset.ncols()), 0.);
        for centroid in centroids.iter() {
            for child_key in centroid.children.iter() {
                let child_point = dataset.slice(s![*child_key,..]);
                
                for i in 0..child_point.len() {
                    residuals[[*child_key, i]] =  child_point[i] - centroid.point[i];
                }
            }
        }

        println!("residuals, shape {:?}", residuals.shape());


        // Residuals PQ Training data
        println!("Residuals PQ Training data");
        let residuals_training_data = self.random_traindata(residuals.view(), 2000);
        println!("residuals_training_data, shape {:?}", residuals_training_data.shape());

        // Train residuals codebook
        let mut residuals_codebook = Array::from_elem((self.m, self.k), Array::zeros(self.sub_dimension));

        for m in 0..self.m {
            let begin = self.sub_dimension * m;
            let end = begin + self.sub_dimension - 1;
            let partial_data = residuals_training_data.slice(s![.., begin..end]);
            let mut pq_kmeans = PQKMeans::new(self.k, self.max_iterations);
            let codewords = pq_kmeans.run(partial_data.view());
            for (k, (centroid,_)) in codewords.iter().enumerate() {
                residuals_codebook[[m,k]] = centroid.to_owned();
            }
        }

        self.residuals_codebook = residuals_codebook;

        // println!("residuals_training_data, shape {:?}", *residuals_codebook.shape());

        // Residuals Encoding
        let  mut pqcodes = Array::from_elem(dataset.nrows(), Array::from_elem(self.m, 0));
        for n in 0..residuals.nrows() {
            for m in 0..self.m {
                let begin = self.sub_dimension * m;
                let end = begin + self.sub_dimension - 1;
                let partial_data = residuals.slice(s![n, begin..end]);

                let mut best_centroid = 0;
                let mut best_distance = f64::NEG_INFINITY;

                for k in 0..self.k {
                    let centroid = &self.residuals_codebook[[m,k]];
                    let distance = distance::cosine_similarity(&(centroid).view(), &partial_data);
                    if best_distance < distance {
                        best_centroid = k;
                        best_distance = distance;
                    }
                }
                pqcodes[n][m] = best_centroid;
            }
        }

        
        let mut coarse_quantizer = Vec::<PQCentroid>::with_capacity(self.m);
        for centroid in centroids.iter() {
            let mut pqchilderen =  HashMap::<usize, Vec::<usize>>::new();
            for child in centroid.children.iter() {
                let codes = &pqcodes[*child];
                pqchilderen.insert(*child, codes.to_vec());
            }
            let pqc = PQCentroid{id: centroid.id, point: centroid.point.to_owned(), children: pqchilderen};
            coarse_quantizer.push(pqc);
        }
        
        self.coarse_quantizer = coarse_quantizer;
        // println!("coarse_quantizer, shape {:?}", coarse_quantizer.len());
    }
    
}



impl AlgorithmImpl for ProductQuantization {

    fn __str__(&self) {
        self.name.to_string();
    }

    fn fit(&mut self, dataset: ArrayView2::<f64>) {
        self.dimension = dataset.slice(s![0,..]).len();
        self.sub_dimension = self.dimension / self.m;
        // self.dataset = Some(dataset.to_owned());
        
        // // Create random selected train data from dataset
        // let train_data = self.random_traindata(dataset, self.training_size);
        // if self.verbose_print {
        //     println!("Training data created shape: {:?}", train_data.shape());
        // }

        // // Create codebook, [m,k,[sd]] m-th subspace, k-th codewords, sd-th dimension
        // // Compute codebook from training data using k-means.
        // self.train_codebook(train_data.view());
        // if self.verbose_print {
        //     println!("Codebook created [m, k, d], shape: {:?}", self.codebook.as_ref().unwrap().shape());
        // }

        // // println!("CODEBOOK: \n {:?}", self.codebook);

        // // Compute PQ Codes
        // self.pqcodes = Some(self.dataset_to_pqcodes(dataset));
        // if self.verbose_print {
        //     println!("PQ Codes computed, shape {:?}", self.pqcodes.as_ref().unwrap().shape());
        // }

        // println!("PQ CODES: \n {:?}", self.pqcodes);
        self.invertedList(dataset);
    }

    fn query(&self, query: &ArrayView1::<f64>, result_count: u32) -> Vec<usize> {

        // Find best coarse_quantizer
        let mut best_match: (f64, usize) = (f64::NEG_INFINITY, 0);
        for centroid in self.coarse_quantizer.iter() {
            let distance = distance::cosine_similarity(&centroid.point.view() , &query);
            if best_match.0 < distance { best_match = (distance, centroid.id); }
        }

        println!("Best coarse_quantizer to search in {}", best_match.1);

        // Lets find matches in best coarse_quantizer
        let bcq = &self.coarse_quantizer[best_match.1];
        
        // Compute residuals between query and coarse_quantizer
        let mut rq = Array::from_elem(query.len(), 0.);
        for i in 0..query.len() {
            rq[i] = query[i] - bcq.point[i];
        }

        // Compute pq codes for query residuals and get values from codebook
        let mut rq_pq_codes = Array::from_elem(self.m, 0);
        for m in 0..self.m {
            let begin = self.sub_dimension * m;
            let end = begin + self.sub_dimension - 1;
            let partial_data = rq.slice(s![begin..end]);

            let mut best_centroid = 0;
            let mut best_distance = f64::NEG_INFINITY;

            for k in 0..self.k {
                let centroid = &self.residuals_codebook[[m,k]];
                let distance = distance::cosine_similarity(&(centroid).view(), &partial_data);
                if best_distance < distance {
                    best_centroid = k;
                    best_distance = distance;
                }
            }
            rq_pq_codes[m] = best_centroid;
        }
        println!("rq_pq_codes \n{:?}", rq_pq_codes);

        let mut pq_res_values = Array::from_elem(self.m, Array::from_elem(self.sub_dimension, 0.));
        let mut rq_point = Vec::<f64>::new();
        for (m, k) in rq_pq_codes.iter().enumerate() {
            let blah = &self.residuals_codebook[[m, *k]];
            pq_res_values[m] = blah.to_owned();
            for b in blah.iter() {
                rq_point.push(*b)
            }   
        }
        let arqpoint = Array::from(rq_point);
        println!("pq_res_values \n{:?}", pq_res_values);

        let mut best_match: (f64, usize) = (f64::NEG_INFINITY, 0);
        for child in bcq.children.iter() {
            let mut point = Array::from_elem(self.m, Array::from_elem(self.sub_dimension, 0.));
            let mut c_point = Vec::<f64>::new();
            for (m, k) in child.1.iter().enumerate() {
                let blah = &self.residuals_codebook[[m, *k]];
                point[m] = blah.to_owned();
                for b in blah.iter() {
                    c_point.push(*b)
                }   
            }
            let acpoint = Array::from(c_point);
            let distance = distance::cosine_similarity(&arqpoint.view() , &acpoint.view());
            if best_match.0 < distance { best_match = (distance, *child.0); }
        }
        println!("best_match!!! {}", best_match.1);
                
        
        // for (index, pq_residuals) in bcq.children.iter() {
        //     println!("index {} pq_residuals\n{:?}", index, pq_residuals);
        //     let ri = Array1::from_elem(pq_residuals.len(), 0.);
        //     for i in 0..pq_residuals.len() {
        //         ri[i] = self.residuals_codebook[]
        //     }
        // }
        

        // let lookfor = vec![97478, 262700, 846101, 671078, 232287, 727732, 544474, 1133489, 723915, 660281];
        
        // let mut best_candidates = BinaryHeap::new();
        // for i in 0..dists.len() {
        //     if lookfor.contains(&(i as i32)) {
        //         println!("found {} with value {}", i, dists[i]);
        //     }
        //     best_candidates.push(DataEntry{index: i, distance: dists[i]});
        // }

        let mut best_n_candidates: Vec<usize> = Vec::new();
        // for _ in 0..result_count {
        //     let cand = best_candidates.pop().unwrap();
        //     println!("Candidate {:?}", cand);
        //     best_n_candidates.push(cand.index);
        // }
        
        best_n_candidates.reverse();
        if self.verbose_print {
            println!("best_n_candidates \n{:?}", best_n_candidates);
        }
        best_n_candidates
    }

}