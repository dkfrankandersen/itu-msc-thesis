use ndarray::{Array1, ArrayView1, ArrayView2, s};
// use crate::algs::data_entry::{DataEntry};
use crate::algs::*;
// use std::collections::BinaryHeap;
use rand::prelude::*;
use std::collections::HashMap;
use colored::*;

#[derive(Clone, PartialEq, Debug)]
pub struct Centroid {
    id: i32,
    pub point: Array1::<f64>,
    pub children: Vec::<usize>
}

impl Centroid {
    fn new(id: i32, point: Array1::<f64>) -> Self {
        Centroid {
            id: id,
            point: point,
            children: Vec::<usize>::new()
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct SubCodebook {
    id: i32,
    pub point: Array1::<f64>,
    pub children: HashMap::<i32, Centroid>
}

impl SubCodebook {
    fn new(id: i32, point: Array1::<f64>) -> Self {
        SubCodebook {
            id: id,
            point: point,
            children: HashMap::<i32, Centroid>::new()
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProductQuantization {
    name: String,
    metric: String,
    dataset: Option<Array2::<f64>>,
    codebook: HashMap::<i32, SubCodebook>,
    clusters: i32,
    subvectors: i32,
    max_iterations: i32,
    clusters_to_search: i32,
    verbose_print: bool
}


impl ProductQuantization {
    pub fn new(verbose_print: bool, clusters: i32, subvectors: i32, max_iterations: i32, clusters_to_search: i32) -> Self {
        ProductQuantization {
            name: "FANN_product_quantization()".to_string(),
            metric: "cosine".to_string(),
            dataset: None,
            codebook: HashMap::<i32, SubCodebook>::new(),
            clusters: clusters,         // K
            subvectors: subvectors,     // M
            max_iterations: max_iterations,
            clusters_to_search: clusters_to_search,
            verbose_print: verbose_print
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

    fn init(&mut self, dataset: &ArrayView2::<f64>) { 
        let mut rng = thread_rng();
        let dist_uniform = rand::distributions::Uniform::new_inclusive(0, self.dataset_size());
        let mut init_k_sampled: Vec<usize> = vec![];
        let data_dimension = dataset.slice(s![0,..]).len();
        let sub_dimensions = data_dimension / self.subvectors as usize;

        for i in 0..self.subvectors {
            let rand_key = rng.sample(dist_uniform);
            init_k_sampled.push(rand_key);
            let candidate = dataset.slice(s![rand_key,..]);
            let mut sub_codebook = SubCodebook::new(i, candidate.to_owned());
            let mut partial_dim = 0;
            loop {
                if partial_dim >= data_dimension {
                    break
                }
                let partial_candidate = candidate.slice(s![partial_dim..partial_dim+sub_dimensions]);
                // println!("partial_candidate {} to {}\n {:?}", d, d+sub_dimensions, partial_candidate);
                partial_dim = partial_dim+sub_dimensions;
                let new_centroid = Centroid::new(partial_dim as i32, partial_candidate.to_owned());
                sub_codebook.children.insert(partial_dim as i32, new_centroid);
            }
            self.codebook.insert(i, sub_codebook);
            
        }

        println!("\n\n CODEBOOK \n{:?}", self.codebook);
    }

    fn assign(&mut self) {
        for (_, sub_codebook) in self.codebook.iter_mut() {
            for (_, centroid) in sub_codebook.children.iter_mut() {
                centroid.children.clear();
            }
        }

        let data_dimension = self.dataset.as_ref().unwrap().slice(s![0,..]).len();
        let sub_dimensions = data_dimension / self.subvectors as usize;

        for (idx, candidate) in self.dataset.as_ref().unwrap().outer_iter().enumerate() {
            let mut best_centroid = -1;
            let mut best_distance = f64::NEG_INFINITY;
            for (&key, centroid) in self.codebook.iter_mut() {
                let distance = distance::cosine_similarity(&(centroid.point).view(), &candidate);
                if best_distance < distance {
                    best_centroid = key;
                    best_distance = distance;
                }
            }
            if best_centroid >= 0 {
                self.codebook.get_mut(&best_centroid).unwrap().children.push(idx);
            }            
        }

        

    }

    fn update(&mut self) { 

    }

    fn run_pq(&mut self, max_iterations: i32, dataset: &ArrayView2::<f64>) {
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

impl AlgorithmImpl for ProductQuantization {

    fn __str__(&self) {
        self.name.to_string();
    }

    fn done(&self) {}

    fn get_memory_usage(&self) {}

    fn fit(&mut self, dataset: ArrayView2::<f64>) {
        self.dataset = Some(dataset.to_owned());
        self.run_pq(self.max_iterations, &dataset);
        
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