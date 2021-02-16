use ndarray::{Array2};
#[path="distance.rs"]
mod distance;

pub fn bruteforce_search(ds_test: Array2<f32>,
    ds_train: Array2<f32>) {

    for (idx_test, test_vector) in ds_test.outer_iter().enumerate() {
    let mut best_dist_euc:f32 = f32::INFINITY;
    let mut best_dist_cos:f32 = f32::NEG_INFINITY;
    let mut best_dist_ang:f32 = f32::INFINITY;
    let mut best_index_euc:usize = 0;
    let mut best_index_cos:usize = 0;
    let mut best_index_ang:usize = 0;
    for (idx_train, train_vector) in ds_train.outer_iter().enumerate() {

    let test_vector_norm = &test_vector;
    let row_vector_norm = &train_vector;

    let dist_euc = distance::dist_euclidian(&test_vector_norm.view(), &row_vector_norm.view());
    if dist_euc < best_dist_euc {
    best_index_euc = idx_train;
    best_dist_euc = dist_euc;
    }

    let dist_cos = distance::dist_cosine_similarity(&test_vector_norm.view(), &row_vector_norm.view());
    if dist_cos > best_dist_cos {
    best_index_cos = idx_train;
    best_dist_cos = dist_cos;
    }

    let dist_ang = distance::dist_angular_similarity(&test_vector_norm.view(), &row_vector_norm.view());
    if dist_ang < best_dist_ang {
    best_index_ang = idx_train;
    best_dist_ang = dist_ang;
    }
    }
    println!("Test index: {}", idx_test);
    println!("EUC best index: {} with dist: {}", best_index_euc, best_dist_euc);
    println!("COS best index: {} with dist: {}", best_index_cos, best_dist_cos);
    println!("ANG best index: {} with dist: {}", best_index_ang, best_dist_ang);
    }
}