use ndarray::{Array1, ArrayView1, ArrayView2, s};
// use crate::algs::data_entry::{DataEntry};
use crate::algs::*;
// use std::collections::BinaryHeap;
// use rand::prelude::*;
use std::collections::HashMap;
// use colored::*;

#[derive(Clone, PartialEq, Debug)]
pub struct Centroid {
    id: i32,
    pub point: Array1::<f64>,
    pub children: Vec::<usize>
}

impl Centroid {
    fn new(id: i32, point: Array1::<f64>) -> Centroid {
        Centroid {
            id: id,
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
    codebook: HashMap::<i32, Centroid>,
    clusters: i32,
    max_iterations: i32,
    clusters_to_search: i32,
    verbose_print: bool
}


impl ProductQuantization {
    pub fn new(verbose_print: bool, clusters: i32, max_iterations: i32, clusters_to_search: i32) -> Self {
        ProductQuantization {
            name: "FANN_product_quantization()".to_string(),
            metric: "cosine".to_string(),
            dataset: None,
            codebook: HashMap::<i32, Centroid>::new(),
            clusters: clusters,
            max_iterations: max_iterations,
            clusters_to_search: clusters_to_search,
            verbose_print: verbose_print
        }
    }

    fn init(&mut self, dataset: &ArrayView2::<f64>) { 

    }

    fn assign(&mut self) { 

    }

    fn update(&mut self) { 

    }

    fn run_pq(&mut self, max_iterations: i32, dataset: &ArrayView2::<f64>) {
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