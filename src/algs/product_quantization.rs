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

#[derive(Debug, Clone)]
pub struct ProductQuantization {
    name: String,
    metric: String,
    dataset: Option<Array2::<f64>>,
    codebook: Vec<Vec<Centroid>>,
    pqcodes: Option::<Array2::<usize>>,
    m: usize,
    training_size: usize,
    k: usize,
    max_iterations: usize,
    clusters_to_search: usize,
    verbose_print: bool,
    dimension: usize,
    sub_dimension: usize
}


impl ProductQuantization {
    pub fn new(verbose_print: bool, m: usize, training_size: usize, k: usize, max_iterations: usize, clusters_to_search: usize) -> Self {
        ProductQuantization {
            name: "FANN_product_quantization()".to_string(),
            metric: "angular".to_string(),
            dataset: None,
            codebook: Vec::with_capacity(m),
            pqcodes: None,
            m: m,         // M
            training_size: training_size,
            k: k,         // K
            max_iterations: max_iterations,
            clusters_to_search: clusters_to_search,
            verbose_print: verbose_print,
            dimension: 0,
            sub_dimension: 0
        }
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
        let dist_uniform = rand::distributions::Uniform::new_inclusive(0, dataset.nrows());
        let mut init_k_sampled: Vec<usize> = vec![];


        /*
            What we want is:
                1. Uniform sample M candidates
                2. For each datapoint split into partial_candidates with partial dimension 
                3. Init sub codebooks for each partial dimension as k-means
                4. 
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

        println!("\n\n CODEBOOK \n{:?}", self.codebook);
    }

    fn assign(&mut self) {

    }

    fn update(&mut self) { 

    }

    fn run_pq(&mut self, max_iterations: usize, dataset: &ArrayView2::<f64>) {
        self.init(dataset);
        // loop {
        //     if self.verbose_print && (iterations == 1 || iterations % 10 == 0) {
        //         println!("Iteration {}", iterations);
        //     }
        //     if iterations > max_iterations {
        //         if self.verbose_print {
        //             println!("Max iterations reached, iterations: {}", iterations-1);
        //         }
        //         break;
        //     } else if self.codebook == last_codebook {
        //         if self.verbose_print {
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
    k: usize,
    max_iterations: usize,
    codebook: Vec::<(Array1::<f64>, Vec::<usize>)>,
    verbose_print: bool
}

impl KMeans {

    pub fn new(k: usize, max_iterations: usize) -> Self {
        KMeans{
            k: k,
            max_iterations: max_iterations,
            codebook: Vec::with_capacity(k),
            verbose_print: true
        }
    }

    fn print_codebook(&self, info: &str, codebook: &Vec::<(Array1::<f64>, Vec::<usize>)>) {
        println!("{}", info.to_string().on_white().black());
        for (k, (centroid, children)) in codebook.iter().enumerate() {
            println!("-> centroid C{:?} |  children: {:?} | point sum: {:?}", k, children.len(), centroid.sum());
        }
    }

    pub fn create_random_traindata(dataset: ArrayView2::<f64>, train_dataset_size: usize) -> Array2::<f64> {
        let mut rng = rand::thread_rng();
        let range = Uniform::new(0 as usize, dataset.nrows() as usize);
        let random_datapoints: Vec<usize> = (0..train_dataset_size).map(|_| rng.sample(&range)).collect();
        println!("Random datapoints [{}] for training, between [0..{}]", train_dataset_size, dataset.nrows());
        
        let mut train_data = Array2::zeros((train_dataset_size, dataset.ncols()));
        for (i,v) in random_datapoints.iter().enumerate() {
            let data_row = dataset.slice(s![*v,..]);
            train_data.row_mut(i).assign(&data_row);
    }
        println!("Train data ready, shape: {:?}", train_data.shape());
        // println!("train_data:\n{}", train_data);
        return train_data;
    }

    pub fn run(&mut self, dataset: ArrayView2::<f64> ) -> &Vec::<(Array1::<f64>, Vec::<usize>)> {

        self.codebook = Vec::with_capacity(self.k);
        self.init(dataset);
        // self.print_codebook("Codebook after init: ", &self.codebook);
        
        let mut last_codebook = Vec::with_capacity(self.k);
        let mut iterations = 1;
        loop {
            if self.verbose_print && (iterations == 1 || iterations % 10 == 0) {
                println!("Iteration {}", iterations);
            }
            if iterations > self.max_iterations {
                if self.verbose_print {
                    println!("Max iterations reached, iterations: {}", iterations-1);
                }
                break;
            } else if self.codebook == last_codebook {
                if self.verbose_print {
                    println!("Computation has converged, iterations: {}", iterations-1);
                }
                break;
            }
    
            last_codebook = self.codebook.clone();

            self.assign(dataset);
            self.update(dataset);
            iterations += 1;
        }
        self.print_codebook("Codebook after run: ", &self.codebook);
        return &self.codebook;
    }

    fn init(&mut self, dataset: ArrayView2::<f64>) {
        let mut rng = thread_rng();
        let dist_uniform = rand::distributions::Uniform::new_inclusive(0, dataset.nrows()-1);
        for i in 0..self.k {
            let rand_key = rng.sample(dist_uniform);
            let candidate = dataset.slice(s![rand_key,..]);
            self.codebook.push((candidate.to_owned(), Vec::<usize>::new()));
        }

        if self.verbose_print {
            self.print_codebook("Codebook after init: ", &self.codebook);
        }      
    }

    fn assign(&mut self, dataset: ArrayView2::<f64>) {
        for (_,children) in self.codebook.iter_mut() {
            children.clear();
        }

        for (idx, candidate) in dataset.outer_iter().enumerate() {
            let mut best_centroid = None;
            let mut best_distance = f64::NEG_INFINITY;
            for (k, centroid) in self.codebook.iter().enumerate() {
                let distance = distance::cosine_similarity(&(centroid.0).view(), &candidate);
                if best_distance < distance {
                    best_centroid = Some(k);
                    best_distance = distance;
                }
            }
            if best_centroid.is_some() {
                self.codebook[best_centroid.unwrap()].1.push(idx);
            } 
        }
    }

    fn update(&mut self, dataset: ArrayView2::<f64>) {
        for (centroid, childeren) in self.codebook.iter_mut() {
            if childeren.len() > 0 {
                for i in 0..centroid.len() {
                    centroid[i]= 0.;
                }
                
                for child_key in childeren.iter() {
                    let child_point = dataset.slice(s![*child_key,..]);
                    for (i, x) in child_point.iter().enumerate() {
                        centroid[i] += x;
                    }
                }
    
                for i in 0..centroid.len() {  
                    centroid[i] = centroid[i]/childeren.len() as f64;
                }
            }
        }
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
        // Create codebook, [m,k,[sd]] m-th subspace, k-th codewords, sd-th dimension
        let mut codebook = Array::from_elem((self.m, self.k), Array::zeros(self.sub_dimension));
        println!("Codebook created [m, k, d], shape: {:?}", codebook.shape());
        // ######################################################################
        // Create random selected train data from dataset
        let train_data = KMeans::create_random_traindata(dataset, self.training_size);

        // ######################################################################
        // Compute codebook from training data using k-means.
        for m in 0..self.m {
            let begin = self.sub_dimension * m;
            let end = begin + self.sub_dimension - 1;
            let partial_data = train_data.slice(s![.., begin..end]);
            if self.verbose_print {
                println!("Run k-means for m [{}], sub dim {:?}, first element [{}]", m, partial_data.shape(), partial_data[[0,0]]);
            }
            let mut kmeans = KMeans::new(self.k, self.max_iterations);
            let codewords = kmeans.run(partial_data.view());
            for (k, (centroid,_)) in codewords.iter().enumerate() {
                    codebook[[m,k]] = centroid.to_owned();
            }
            if self.verbose_print {
                // println!("Codebook for m {} childeren: {:?} \n{:?}", m, codebook_for_m[0].1.len(), codebook_for_m[0]);
            }
        }

        // ######################################################################
        // Compute PQ Codes
        let mut pq_code = Array::from_elem((self.m, dataset.nrows()), 0);
        for idx in 0..dataset.nrows() {
            for m in 0..self.m {
                let begin = self.sub_dimension * m;
                let end = begin + self.sub_dimension - 1;
                let partial_data = dataset.slice(s![idx, begin..end]);


                let mut best_centroid: Option::<usize> = None;
                let mut best_distance = f64::NEG_INFINITY;

                for k in 0..self.k {
                    let centroid = &codebook[[m,k]];
                    let distance = distance::cosine_similarity(&(centroid).view(), &partial_data);
                    if best_distance < distance {
                        best_centroid = Some(k);
                        best_distance = distance;
                    }
                }
                if best_centroid.is_some() {
                    pq_code[[m, idx]] = best_centroid.unwrap();
                } 
            }   
        }

        println!("{:?}", pq_code.column(5));
        self.pqcodes = Some(pq_code);

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