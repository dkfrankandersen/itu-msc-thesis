use ndarray::{ArrayView1, ArrayView2};
use crate::algs::distance;
use crate::algs::data_entry::{DataEntry};
use crate::algs::*;
use std::collections::BinaryHeap;

#[derive(Debug, Clone)]
pub struct Bruteforce {
    name: String,
    metric: String,
    dataset: Option<Array2::<f64>>,
    verbose_print: bool
}

impl Bruteforce {
    pub fn new() -> Self {
        Bruteforce {
            name: "FANN_bruteforce()".to_string(),
            metric: "cosine".to_string(),
            dataset: None,
            verbose_print: false
        }
    }
}

impl AlgorithmImpl for Bruteforce {

    fn __str__(&self) {
        self.name.to_string();
    }

    fn done(&self) {}

    fn get_memory_usage(&self) {}

    fn fit(&mut self, dataset: ArrayView2::<f64>) {
        self.dataset = Some(dataset.to_owned());
    }

    fn batch_query(&self) {}

    fn get_batch_results(&self) {}
    
    fn get_additional(&self) {
        
    }
    
    fn query(&self, p: &ArrayView1::<f64>, result_count: u32) -> Vec<usize> {
        let mut best_candidates = BinaryHeap::new();
        if self.dataset.is_none() {
            println!("Dataset missing");
        }
        for (idx, candidate) in self.dataset.as_ref().unwrap().outer_iter().enumerate() {
            let dist = distance::cosine_similarity(&p, &candidate);
            if best_candidates.len() < result_count as usize {
                best_candidates.push(DataEntry {
                    index: idx,  
                    distance: -dist
                });
                
            } else {
                let min_val: DataEntry = *best_candidates.peek().unwrap();
                if dist > -min_val.distance {
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