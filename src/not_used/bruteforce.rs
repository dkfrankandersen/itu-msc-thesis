use ndarray::{Array2, ArrayView1, ArrayView2};
use crate::algs::distance;
use crate::algs::pq;
use std::collections::BinaryHeap;

pub fn bruteforce_search(v: &ArrayView1::<f64>,
                            dataset: &ArrayView2::<f64>,
                            dist_type: distance::DistType) -> (usize, f64) {

    let mut best_dist: f64;
    let mut best_index: usize = 0;

    match dist_type {
        distance::DistType::Angular     => best_dist = f64::INFINITY,
        distance::DistType::Cosine      => best_dist = f64::NEG_INFINITY,
        distance::DistType::Euclidian   => best_dist = f64::INFINITY,
    }
    
    for (idx, candidate) in dataset.outer_iter().enumerate() {
        let dist: f64;
        match dist_type {
            distance::DistType::Angular     => {
                dist = distance::euclidian(&v.view(), &candidate.view());
                if dist < best_dist {
                    best_index = idx;
                    best_dist = dist;
                };
            },
            distance::DistType::Cosine      => {
                dist = distance::cosine_similarity(&v.view(), &candidate.view());
                if dist > best_dist {
                    best_index = idx;
                    best_dist = dist;
                }
            },
            distance::DistType::Euclidian   => {
                dist = distance::angular_similarity(&v.view(), &candidate.view());
                if dist < best_dist {
                    best_index = idx;
                    best_dist = dist;
                }
            },
        }
    }
        println!("Best index: {} with dist: {}", best_index, best_dist);
        (best_index, best_dist)
}

pub fn search_dataset(ds_test: &Array2<f64>, ds_train: &Array2<f64>, no_of_results: usize) {

    for (idx_test, test_vector) in ds_test.outer_iter().enumerate() {
        let mut best_dist_euc:f64 = f64::INFINITY;
        let mut best_dist_cos:f64 = f64::NEG_INFINITY;
        let mut best_dist_ang:f64 = f64::INFINITY;
        let mut best_index_euc:usize = 0;
        let mut best_index_cos:usize = 0;
        let mut best_index_ang:usize = 0;
        for (idx_train, train_vector) in ds_train.outer_iter().enumerate() {
            let test_vector_norm = &test_vector;
            let row_vector_norm = &train_vector;

            let dist_euc = distance::euclidian(&test_vector_norm.view(), &row_vector_norm.view());
            if dist_euc < best_dist_euc {
                best_index_euc = idx_train;
                best_dist_euc = dist_euc;
            }

            let dist_cos = distance::cosine_similarity(&test_vector_norm.view(), &row_vector_norm.view());
            if dist_cos > best_dist_cos {
                best_index_cos = idx_train;
                best_dist_cos = dist_cos;
            }

            let dist_ang = distance::angular_similarity(&test_vector_norm.view(), &row_vector_norm.view());
            if dist_ang < best_dist_ang {
                best_index_ang = idx_train;
                best_dist_ang = dist_ang;
            }
        }
        println!("Test index: {}", idx_test);
        println!("EUC best index: {} with dist: {}", best_index_euc, best_dist_euc);
        println!("COS best index: {} with dist: {}", best_index_cos, best_dist_cos);
        println!("ANG best index: {} with dist: {}", best_index_ang, best_dist_ang);
        
        if idx_test >= no_of_results {
            break;
        }
    }
}