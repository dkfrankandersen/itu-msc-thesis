use ndarray::{Array2, ArrayView1, ArrayView2};
use crate::algs::distance;
use crate::algs::pq;
use crate::algs::*;
use std::collections::BinaryHeap;

struct Bruteforce {
    name: String
}

impl AlgorithmImpl for Bruteforce {

    fn new(&self, name: String) -> Self {
        Bruteforce {
            name: name
        }
    }

    fn __str__(&self) {
        self.name;
    }

    fn done(&self) {}

    fn get_memory_usage(&self) {}

    fn fit(&self) {}

    fn batch_query(&self) {}

    fn get_batch_results(&self) {}
    
    fn get_additional(&self) {
        
    }
    
    fn query(&self, p: &ArrayView1::<f64>, dataset: &ArrayView2::<f64>, result_count: u32) -> Vec<usize> {
        let mut best_candidates = BinaryHeap::new();
        for (idx, candidate) in dataset.outer_iter().enumerate() {
            let dist = distance::cosine_similarity(&p, &candidate);
            if best_candidates.len() < result_count as usize {
                best_candidates.push(pq::DataEntry {
                    index: idx,  
                    distance: -dist
                });
                
            } else {
                let min_val: pq::DataEntry = *best_candidates.peek().unwrap();
                if dist > -min_val.distance {
                    best_candidates.pop();
                    best_candidates.push(pq::DataEntry {
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