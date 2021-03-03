use ndarray::{Array2, ArrayView1, ArrayView2, s};
use crate::algs::distance;
use crate::algs::pq;
use std::collections::BinaryHeap;
use rand::prelude::*;
use std::collections::HashMap;

pub fn query(p: &ArrayView1::<f64>, dataset: &ArrayView2::<f64>, result_count: u32) -> Vec<usize> {

    /*
        Init
        Assign
        Update
        Repeat until convergence or some iteration count

        Repeat X times, select best based on cluster density
    */

    // 1. Init
    let k = 10;
    let max_iterations = 100;
    let n = &dataset.shape()[0]; // shape of rows, cols (vector dimension)
    let mut rng = thread_rng();
    let dist_uniform = rand::distributions::Uniform::new_inclusive(0, n);
    let mut init_k_sampled: Vec<usize> = vec![];
    let mut codebook = HashMap::new();
    let mut codebook_cluster = HashMap::new();
    
    for _ in 0..k {
        let rand_key = rng.sample(dist_uniform);
        init_k_sampled.push(rand_key);
        codebook.insert(k, Vec::<usize>::new());
        let centroid = dataset.slice(s![rand_key,..]);
        codebook_cluster.insert(k, centroid);
    }

    println!("Dataset lenght: {}", n);
    println!("Init k-means with centroids: {:?}", init_k_sampled);
    println!("Centroids: {:?}", codebook_cluster);

    
    // 2. Assign
    println!("Let look for my favorit centroid!");
    for (idx, candidate) in dataset.outer_iter().enumerate() {
        let mut best_centroid = 0;
        let mut best_distance = f64::INFINITY;
        for (&key,_) in codebook.iter() {
            let centroid = &dataset.slice(s![key,..]);
            let distance = distance::cosine_similarity(&centroid, &candidate);
            if best_distance > distance {
                best_centroid = key;
                best_distance = distance;

            }
        }
        codebook.get_mut(&best_centroid).unwrap().push(idx);
        // println!("Assign {} to centroid {} ", idx, best_centroid);

        if idx > 100 {
            break;
        }
    }

    // 3. Update
    
    for (&key,vec) in codebook_cluster.iter() {
        let local_minima = &dataset.slice(s![idx,..]);
        let local_minima = &dataset.slice(s![idx,..]);
        let mut local_minima;
        // for idx in codebook.get(&key) {
        //     if idx == 0 as usize {
        //         let local_minima = &dataset.slice(s![idx,..]);
        //     } else {
        //         let point = &dataset.slice(s![idx,..]);
        //     }
        // }
    }
    let mut codebook = HashMap::new();
    let mut codebook_cluster = HashMap::new();

    // 4. Repeat   

    println!("{:?}", codebook);

    let mut best_n_candidates: Vec<usize> = Vec::new();
    // for _ in 0..result_count {
    //     let idx = (Some(best_candidates.pop()).unwrap()).unwrap();
    //     best_n_candidates.push(idx.index);
    // }
    best_n_candidates
}