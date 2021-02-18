use ndarray::{ArrayView1, ArrayView2};
use crate::algs::distance;
use crate::algs::pq;
use std::collections::BinaryHeap;

pub fn single_search(test_vector: &ArrayView1::<f64>,
                        ds_train: &ArrayView2::<f64>,
                        no_of_matches: i32, dist_type: &str) -> Vec<(usize, f64)> {
    
    let mut heap = BinaryHeap::new();
    for (idx_train, train_vector) in ds_train.outer_iter().enumerate() {
        let dist: f64;
        if dist_type == "angular" {
            dist = distance::dist_angular_similarity(&test_vector, &train_vector);
        } else {
            dist = distance::dist_cosine_similarity(&test_vector, &train_vector);
        }
    
        heap.push(pq::DataEntry {index: idx_train,  distance: dist});
    }

    let mut result: Vec<(usize, f64)> = Vec::new();

    for _ in 0..no_of_matches {
        let idx = (Some(heap.pop()).unwrap()).unwrap();
        result.push((idx.index, idx.distance));
    }
    result
}