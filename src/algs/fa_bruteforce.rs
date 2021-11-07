use ndarray::{ArrayView1, ArrayView2};
use std::collections::BinaryHeap;
pub use ordered_float::*;
use crate::algs::*;
use crate::algs::distance::{DistanceMetric, min_distance};

#[derive(Debug, Clone)]
pub struct FABruteforce {
    name: String,
    metric: String,
    verbose_print: bool,
    dist_metric: DistanceMetric,
}

impl FABruteforce {
    pub fn new(verbose_print: bool, dist_metric: DistanceMetric) -> Result<Self, String> {
        Ok(FABruteforce {
            name: "fa_bruteforce_TR11".to_string(),
            metric: "angular".to_string(),
            verbose_print,
            dist_metric,
        })
    }
}

impl AlgorithmImpl for FABruteforce {

    fn name(&self) -> String {
        self.name.to_string()
    }
    
    fn fit(&mut self, _dataset: &ArrayView2::<f64>) {
    }
    
    fn query(&self, dataset: &ArrayView2::<f64>, query: &ArrayView1::<f64>, results_per_query: usize, _arguments: &[usize]) -> Vec<usize> {
        let mut best_candidates = BinaryHeap::<(OrderedFloat::<f64>, usize)>::new();
        for (index, datapoint) in dataset.outer_iter().enumerate() {
            let distance = OrderedFloat(min_distance(&query, &datapoint, &self.dist_metric));
            if best_candidates.len() < results_per_query {
                best_candidates.push((distance, index));
                
            } else if distance < best_candidates.peek().unwrap().0 {
                    best_candidates.pop();
                    best_candidates.push((distance, index));
            }
        }

        // Pop all candidate indexes from heap and reverse list.
        let mut best_n_candidates: Vec<usize> =  (0..best_candidates.len())
                                                    .map(|_| best_candidates
                                                    .pop().unwrap().1).collect();
        best_n_candidates.reverse();
        best_n_candidates
    }
}