use ndarray::{Array2, ArrayView1, ArrayView2, s};
use crate::algs::distance;
use crate::algs::pq;
use std::collections::BinaryHeap;
use rand::prelude::*;
use std::collections::HashMap;

pub fn query(p: &ArrayView1::<f64>, dataset: &ArrayView2::<f64>, result_count: u32) -> Vec<usize> {

    // 1. Init
    let k = 10;
    let n = &dataset.shape()[0]; // shape of rows, cols (vector dimension)
    let mut rng = thread_rng();
    let distr = rand::distributions::Uniform::new_inclusive(0, n);
    let mut init_k_sampled: Vec<usize> = vec![];
    let mut codebook = HashMap::new();
    for _ in 0..k {
        let rand_key = rng.sample(distr);
        init_k_sampled.push(rand_key);
        codebook.insert(rand_key, Vec::<usize>::new());
    }

    println!("Dataset lenght: {}", n);
    println!("Init k-means with centroids: {:?}", init_k_sampled);
    
    // 2. Assign
    // 3. Update
    // 4. Repeat   

    println!("Let look for my favorit centroid!");
    
    for (idx, candidate) in dataset.outer_iter().enumerate() {
        let mut best_centroid = 0;
        let mut best_distance = f64::INFINITY;
        for (&key, val) in &codebook {
            let centroid = &dataset.slice(s![key,..]);
            let distance = distance::cosine_similarity(&centroid, &candidate);
            if best_distance > distance {
                best_centroid = key;
                best_distance = distance;

            }
        }
        println!("Assign {} to centroid {} ", idx, best_centroid);

    }

    let mut best_n_candidates: Vec<usize> = Vec::new();
    // for _ in 0..result_count {
    //     let idx = (Some(best_candidates.pop()).unwrap()).unwrap();
    //     best_n_candidates.push(idx.index);
    // }
    best_n_candidates
}