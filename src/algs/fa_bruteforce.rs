use ndarray::{ArrayView1, ArrayView2};
use std::collections::BinaryHeap;
pub use ordered_float::*;
use crate::algs::*;
use crate::algs::{distance::{DistanceMetric, cosine_similarity}};

#[derive(Debug, Clone)]
pub struct FABruteforce {
    name: String,
    metric: String,
    verbose_print: bool,
    dist: DistanceMetric,
}

impl FABruteforce {
    pub fn new(verbose_print: bool, dist_metric: DistanceMetric) -> Result<Self, String> {
        return Ok(FABruteforce {
            name: "fa_bruteforce_c01T".to_string(),
            metric: "angular".to_string(),
            verbose_print: verbose_print,
            dist: dist_metric
        });
    }

    pub fn distance(&self, p: &ArrayView1::<f64>, q: &ArrayView1::<f64>) -> f64 {
        -cosine_similarity(&p, &q)
    }

    pub fn min_distance_ordered(&self, p: &ArrayView1::<f64>, q: &ArrayView1::<f64>) -> OrderedFloat::<f64> {
        OrderedFloat(self.distance(p, q))
    }

    pub fn max_distance_ordered(&self, p: &ArrayView1::<f64>, q: &ArrayView1::<f64>) -> OrderedFloat::<f64> {
        OrderedFloat(-self.distance(p, q))
    }
    
}

impl AlgorithmImpl for FABruteforce {

    fn name(&self) -> String {
        self.name.to_string()
    }
    
    fn fit(&mut self, _dataset: &ArrayView2::<f64>) {
        
    }
    
    fn query(&self, dataset: &ArrayView2::<f64>, p: &ArrayView1::<f64>, results_per_query: usize, _arguments: &Vec::<usize>) -> Vec<usize> {
        let mut best_candidates = BinaryHeap::<(OrderedFloat::<f64>, usize)>::new();

        for (idx, datapoint) in dataset.outer_iter().enumerate() {
            let distance = self.min_distance_ordered(&p, &datapoint);
            if best_candidates.len() < results_per_query {
                best_candidates.push((distance, idx));
                
            } else {
                if distance < best_candidates.peek().unwrap().0 {
                    best_candidates.pop();
                    best_candidates.push((distance, idx));
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