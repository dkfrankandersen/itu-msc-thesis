use ndarray::{ArrayView1, ArrayView2};
use std::collections::BinaryHeap;
pub use ordered_float::*;
use crate::algs::*;
use crate::algs::{distance::{DistanceMetric}};

use super::distance::CosineSimilarity;

#[derive(Debug, Clone)]
pub struct FABruteforce {
    name: String,
    metric: String,
    verbose_print: bool,
    dist: DistanceMetric,
    cosine_metric: Option<CosineSimilarity>
}

impl FABruteforce {
    pub fn new(verbose_print: bool, dist_metric: DistanceMetric) -> Result<Self, String> {
        Ok(FABruteforce {
            name: "fa_bruteforce_c05".to_string(),
            metric: "angular".to_string(),
            verbose_print,
            dist: dist_metric,
            cosine_metric: None
        })
    }
}

impl AlgorithmImpl for FABruteforce {

    fn name(&self) -> String {
        self.name.to_string()
    }
    
    fn fit(&mut self, dataset: &ArrayView2::<f64>) {
        println!("Fitting for faster CosineSimilarity");
        self.cosine_metric = Some(CosineSimilarity::new(dataset));
    }
    
    fn query(&self, dataset: &ArrayView2::<f64>, query: &ArrayView1::<f64>, results_per_query: usize, _arguments: &[usize]) -> Vec<usize> {
        let mut best_candidates = BinaryHeap::<(OrderedFloat::<f64>, usize)>::new();
        let cosine_metric = self.cosine_metric.as_ref().unwrap();
        let q_dot_sqrt = cosine_metric.query_dot_sqrt(query);
        for (index, datapoint) in dataset.outer_iter().enumerate() {
            // let distance = OrderedFloat(cosine_similarity(&query, &datapoint));
            let distance = cosine_metric.fast_min_distance_ordered(index, &datapoint, query, q_dot_sqrt);
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