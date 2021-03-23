use ndarray::{Array, Array1, Array2, Array3, ArrayView1, ArrayView2, s};
// use crate::algs::data_entry::{DataEntry};
use crate::algs::*;
// use std::collections::BinaryHeap;
use rand::prelude::*;
use rand::{distributions::Uniform, Rng};
use std::collections::HashMap;
use colored::*;

#[derive(Clone, PartialEq, Debug)]
pub struct Centroid {
    pub point: Array1::<f64>,
    pub children: Vec::<usize>
}

impl Centroid {
    fn new(id: i32, point: Array1::<f64>) -> Self {
        Centroid {
            point: point,
            children: Vec::<usize>::new()
        }
    }
}

// #[derive(Clone, PartialEq, Debug)]
// pub struct SubCodebook {
//     pub point: Array1::<f64>,
//     pub children: Vec<Centroid>
// }

// impl SubCodebook {
//     fn new(point: Array1::<f64>) -> Self {
//         SubCodebook {
//             point: point,
//             children: Vec::new()
//         }
//     }
// }

#[derive(Debug, Clone)]
pub struct ProductQuantization {
    name: String,
    metric: String,
    dataset: Option<Array2::<f64>>,
    codebook: Vec<Vec<Centroid>>,
    k: usize,
    m: usize,
    max_iterations: usize,
    clusters_to_search: usize,
    verbose_print: bool,
    dimension: usize,
    sub_dimension: usize
}


impl ProductQuantization {
    pub fn new(verbose_print: bool, m: usize, k: usize, max_iterations: usize, clusters_to_search: usize) -> Self {
        ProductQuantization {
            name: "FANN_product_quantization()".to_string(),
            metric: "angular".to_string(),
            dataset: None,
            codebook: Vec::with_capacity(m),
            k: k,         // K
            m: m,              // M
            max_iterations: max_iterations,
            clusters_to_search: clusters_to_search,
            verbose_print: verbose_print,
            dimension: 0,
            sub_dimension: 0
        }
    }

    fn dataset_size(&self) -> usize {
        return self.dataset.as_ref().unwrap().shape()[0]; // shape of rows, cols (vector dimension)
    }

    fn print_sum_codebook_children(&self, info: &str, codebook: &HashMap<i32, Centroid>, dataset_len: usize) {
        println!("{}", info.to_string().on_white().black());
        let mut sum = 0;
        for (_, centroid) in codebook.iter() {
            sum += centroid.children.len();
        }
        println!("children: {} == {} dataset points, equal {}", sum.to_string().blue(), dataset_len.to_string().blue(), (sum == dataset_len).to_string().blue());
    }
    
    fn print_codebook(&self, info: &str, codebook: &HashMap<i32, Centroid>) {
        println!("{}", info.to_string().on_white().black());
        for (key, centroid) in codebook.iter() {
            println!("-> centroid C{:?} |  children: {:?} | point sum: {:?}", key, centroid.children.len(), centroid.point.sum());
        }
    }

    fn partial_dimension_from_datapoint(&self, pos: usize,  datapoint: &ArrayView1::<f64>)
    -> Array1<f64> {
        let begin = self.sub_dimension * pos;
        let end = begin + self.sub_dimension - 1;
        return datapoint.slice(s![begin..end]).to_owned();
    }

    fn init(&mut self, dataset: &ArrayView2::<f64>) { 
        let mut rng = thread_rng();
        let dist_uniform = rand::distributions::Uniform::new_inclusive(0, self.dataset_size());
        let mut init_k_sampled: Vec<usize> = vec![];


        /*
            What we want is:
                1. Uniform sample M candidates
                2. For each datapoint split into partial_candidates with partial dimension 
                3. Init sub codebooks for each partial dimension
        */

        // Create M sub codebooks with capacity of k sub centroids
        for i in 0..self.m {
            self.codebook[i] = Vec::with_capacity(self.k);
        }

        // For each codebook perform k-means init
        for (i, sub_codebook) in self.codebook.iter_mut().enumerate() {
            let rand_key = rng.sample(dist_uniform);
            let candidate = dataset.slice(s![rand_key,..]);
            let begin = self.sub_dimension * i;
            let end = begin + self.sub_dimension - 1;
            let partial_candidate = candidate.slice(s![begin..end]).to_owned();

            for (j, centroid) in sub_codebook.iter_mut().enumerate() {
                
                
            }

        }


        // for i in 0..self.m {
        //     let rand_key = rng.sample(dist_uniform);
        //     init_k_sampled.push(rand_key);
        //     let candidate = dataset.slice(s![rand_key,..]);
        //     let mut sub_codebook = SubCodebook::new(i, candidate.to_owned());
        //     let mut partial_dim = 0;
        //     loop {
        //         if partial_dim >= self.dimension {
        //             break
        //         }
        //         let partial_candidate = candidate.slice(s![partial_dim..partial_dim+self.sub_dimension]);
        //         // println!("partial_candidate {} to {}\n {:?}", d, d+sub_dimensions, partial_candidate);
        //         partial_dim = partial_dim+self.sub_dimension;
        //         let new_centroid = Centroid::new(partial_dim as i32, partial_candidate.to_owned());
        //         sub_codebook.children.insert(partial_dim as i32, new_centroid);
        //     }
        //     self.codebook.insert(i, sub_codebook);
            
        // }

        println!("\n\n CODEBOOK \n{:?}", self.codebook);
    }

    fn assign(&mut self) {
        // for (_, sub_codebook) in self.codebook.iter_mut() {
        //     for (_, centroid) in sub_codebook.children.iter_mut() {
        //         centroid.children.clear();
        //     }
        // }

        // for (idx, candidate) in self.dataset.as_ref().unwrap().outer_iter().enumerate() {
        //     let mut best_centroid = -1;
        //     let mut best_distance = f64::NEG_INFINITY;
        //     let mut partial_dim = 0;

        //     loop {
        //         if partial_dim >= self.dimension {
        //             break;
        //         }
        //         let partial_candidate = candidate.slice(s![partial_dim..partial_dim+self.sub_dimension]);

        //         for (&key, centroid) in self.codebook.iter_mut() {
        //             let distance = distance::cosine_similarity(&(centroid.point).view(), &candidate);
        //             if best_distance < distance {
        //                 best_centroid = key;
        //                 best_distance = distance;
        //             }
        //         }
        //         if best_centroid >= 0 {
        //             self.codebook.get_mut(&best_centroid).unwrap().children.push(idx);
        //         }  
        //     }           
        // }
    }

    fn update(&mut self) { 

    }

    fn run_pq(&mut self, max_iterations: usize, dataset: &ArrayView2::<f64>) {
        self.init(dataset);
        // loop {
        //     if self.verbode_print && (iterations == 1 || iterations % 10 == 0) {
        //         println!("Iteration {}", iterations);
        //     }
        //     if iterations > max_iterations {
        //         if self.verbode_print {
        //             println!("Max iterations reached, iterations: {}", iterations-1);
        //         }
        //         break;
        //     } else if self.codebook == last_codebook {
        //         if self.verbode_print {
        //             println!("Computation has converged, iterations: {}", iterations-1);
        //         }
        //         break;
        //     }

        //     self.assign();
        //     self.update();
        //     iterations += 1;
        // }
    }
}


#[derive(Debug, Clone)]
struct KMeans {

}

impl KMeans {
    pub fn run(&self, dataset: &ArrayView2::<f64>, k: usize, max_iterations: usize) -> Array2::<f64> {

        let codebook = 
        self.init(dataset, k);

    }

    fn init(&mut self, dataset: &ArrayView2::<f64>, k: usize) {
        let mut rng = thread_rng();
        let dist_uniform = rand::distributions::Uniform::new_inclusive(0, dataset.nrows());
        for i in 0..k {
            let rand_key = rng.sample(dist_uniform);
            let candidate = dataset.slice(s![rand_key,..]);

        }
    }

    fn assign() {

    }

    fn update() {

    }


}

impl AlgorithmImpl for ProductQuantization {

    fn __str__(&self) {
        self.name.to_string();
    }

    fn done(&self) {}

    fn get_memory_usage(&self) {}

    fn fit(&mut self, dataset: ArrayView2::<f64>) {
        self.dimension = dataset.slice(s![0,..]).len();
        self.sub_dimension = self.dimension / self.m;
        self.dataset = Some(dataset.to_owned());

        
        // ######################################################################
        // Create codebook, [m,k,d] m-th subspace, k-th codewords, d-th dimension
        let codebook = Array3::<f64>::zeros((self.m, self.k, self.sub_dimension));
        println!("Codebook created [m, k, d], shape: {:?}", codebook.shape());
        // ######################################################################


        // ######################################################################
        // Create random selected train data from dataset
        let train_dataset_len = 2000;
        let mut rng = rand::thread_rng();
        let range = Uniform::new(0 as usize, self.dataset_size() as usize);
        let random_datapoints: Vec<usize> = (0..train_dataset_len).map(|_| rng.sample(&range)).collect();
        println!("Random datapoints [{}] for training, between [0..{}]", train_dataset_len, self.dataset_size());
        
        let mut train_data = Array2::zeros((train_dataset_len, self.dimension));
        for (i,v) in random_datapoints.iter().enumerate() {
            let data_row = dataset.slice(s![*v,..]);
            train_data.row_mut(i).assign(&data_row);
        }
        println!("Train data ready, shape: {:?}", train_data.shape());
        // println!("train_data:\n{}", train_data);

        // ######################################################################


        // ######################################################################
        // Compute codebook from training data using k-means.
        for m in 0..self.m {

            let begin = self.sub_dimension * m;
            let end = begin + self.sub_dimension - 1;
            let partial_data = train_data.slice(s![.., begin..end]);
            println!("Run k-means for m [{}], sub dim {:?}, first element [{}]", m, partial_data.shape(), partial_data[[0,0]]);
            
            // let codebook = KMeans(dataset: train_data, clusters: self.k, max_iterations: self.max_iterations);

        }

        

    }

    fn batch_query(&self) {}

    fn get_batch_results(&self) {}
    
    fn get_additional(&self) {
        
    }

    fn query(&self, p: &ArrayView1::<f64>, result_count: u32) -> Vec<usize> {
        
        let mut best_n_candidates: Vec<usize> = Vec::new();
        best_n_candidates.reverse();
        if self.verbose_print {
            println!("best_n_candidates \n{:?}", best_n_candidates);
        }
        best_n_candidates
    }

}