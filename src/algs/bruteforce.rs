use ndarray::{ArrayView1, ArrayView2};
use std::collections::BinaryHeap;
pub use ordered_float::*;
use crate::algs::*;
//use crate::util::{DebugTimer};

#[derive(Debug, Clone)]
pub struct Bruteforce {
    name: String,
    metric: String,
    verbose_print: bool
}

impl Bruteforce {
    pub fn new(verbose_print: bool) -> Self {
        Bruteforce {
            name: "FANN_bruteforce()".to_string(),
            metric: "angular".to_string(),
            verbose_print: verbose_print
        }
    }
}

impl AlgorithmImpl for Bruteforce {

    fn name(&self) -> String {
        self.name.to_string()
    }
    
    fn fit(&mut self, _dataset: &ArrayView2::<f64>) {
    }
    
    fn query(&self, dataset: &ArrayView2::<f64>, p: &ArrayView1::<f64>, results_per_query: usize, arguments: &Vec::<usize>) -> Vec<usize> {

        let mut best_candidates = BinaryHeap::<(OrderedFloat::<f64>, usize)>::new();

        for (idx, candidate) in dataset.outer_iter().enumerate() {
            let distance = distance::cosine_similarity(&p, &candidate);
            if best_candidates.len() < results_per_query {
                best_candidates.push((OrderedFloat(-distance), idx));
                
            } else {
                if OrderedFloat(distance) > -best_candidates.peek().unwrap().0 {
                    best_candidates.pop();
                    best_candidates.push((OrderedFloat(-distance), idx));
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