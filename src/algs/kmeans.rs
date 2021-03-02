use ndarray::{Array2, ArrayView1, ArrayView2};
use crate::algs::distance;
use crate::algs::pq;
use std::collections::BinaryHeap;
use rand::prelude::*;

pub fn query(p: &ArrayView1::<f64>, dataset: &ArrayView2::<f64>, result_count: u32) -> Vec<usize> {

    // 1. Init
    let k = 10;
    let n = dataset.len();
    let mut rng = thread_rng();
    let distr = rand::distributions::Uniform::new_inclusive(0, n);
    let mut init_k_sampled: Vec<usize> = vec![];
    for _ in 0..k {
        let rand_val = rng.sample(distr);
        init_k_sampled.push(rand_val);
    }
    println!("Init k-means with centroids: {:?}", init_k_sampled);
    
    // 2. Assign
    // 3. Update
    // 4. Repeat
    
    let mut best_candidates = BinaryHeap::new();
    for (idx, candidate) in dataset.outer_iter().enumerate() {
        best_candidates.push(pq::DataEntry {
                                                index: idx,  
                                                distance: distance::cosine_similarity(&p, &candidate)
                                            });
    }

    let mut best_n_candidates: Vec<usize> = Vec::new();
    for _ in 0..result_count {
        let idx = (Some(best_candidates.pop()).unwrap()).unwrap();
        best_n_candidates.push(idx.index);
    }
    best_n_candidates
}