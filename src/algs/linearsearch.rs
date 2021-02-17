use ndarray::{ArrayView1, ArrayView2};
use crate::algs::distance::dist_cosine_similarity;
use crate::algs::pq;
use std::collections::BinaryHeap;

pub fn single_search(test_vector: &ArrayView1::<f64>,
                        ds_train: &ArrayView2::<f64>,
                        no_of_matches: i32) -> Vec<usize> {
    
    let mut heap = BinaryHeap::new();
    heap.push(pq::DataEntry {index: 2,  distance: 0.00000001});

    let mut best_dist: f64 = f64::NEG_INFINITY;
    let mut best_index: usize = 0;
    
    for (idx_train, train_vector) in ds_train.outer_iter().enumerate() {
        let dist = dist_cosine_similarity(&test_vector, &train_vector);
        heap.push(pq::DataEntry {index: idx_train,  distance: dist});
        if dist > best_dist {
            best_index = idx_train;
            best_dist = dist;
        }
    }
    

    let mut result = Vec::new();

    for _ in 0..no_of_matches {
        let idx = (Some(heap.pop()).unwrap()).unwrap();
        // println!("{:?}", idx.index);
        result.push(idx.index);
    }
    println!("best_index: {:?}", best_index);
    result
}