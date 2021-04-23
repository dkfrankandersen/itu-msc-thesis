use ndarray::{ArrayView1, ArrayView2};
use crate::algs::distance;
use crate::algs::data_entry::{DataEntry};
use crate::algs::*;
use std::collections::BinaryHeap;
extern crate ordered_float;
pub use ordered_float::*;

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

    fn __str__(&self) {
        self.name.to_string();
    }

    fn fit(&mut self, _dataset: &ArrayView2::<f64>) {
    }
    
    fn query(&self, dataset: &ArrayView2::<f64>, p: &ArrayView1::<f64>, result_count: usize) -> Vec<usize> {
        let mut best_candidates = BinaryHeap::new();

        for (idx, candidate) in dataset.outer_iter().enumerate() {
            let dist = distance::cosine_similarity(&p, &candidate);
            if best_candidates.len() < result_count {
                best_candidates.push(DataEntry {
                    index: idx,  
                    distance: -dist
                });
                
            } else {
                let min_val: DataEntry = *best_candidates.peek().unwrap();
                if OrderedFloat(dist) > OrderedFloat(-min_val.distance) {
                    best_candidates.pop();
                    best_candidates.push(DataEntry {
                        index: idx,  
                        distance: -dist
                    });
                }
            }
        }

        let mut best_n_candidates: Vec<usize> = Vec::new();
        for _ in 0..best_candidates.len() {
            let idx = (Some(best_candidates.pop()).unwrap()).unwrap();
            best_n_candidates.push(idx.index);
        }
        best_n_candidates.reverse();
        best_n_candidates
    }
}