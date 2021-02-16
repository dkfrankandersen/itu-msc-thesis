extern crate ndarray;
extern crate hdf5;
use std::time::{Instant};
use ndarray::{ArrayBase, Array2, ViewRepr, Dim};
use priority_queue::PriorityQueue;
mod distance;
mod dataset;

fn bruteforce_search(ds_test: Array2<f32>,
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

fn main() {
    let _e = hdf5::silence_errors();
    let _filename = "datasets/glove-100-angular.hdf5";
    let file = &dataset::get_dataset(_filename);
    
    let ds_train = dataset::get_dataset_f32(file, "train");
    let _ds_train = dataset::normalize_all(ds_train);
    
    let ds_test = dataset::get_dataset_f32(file, "test");
    let _ds_test = dataset::normalize_all(ds_test);

    let ds_distance = dataset::get_dataset_f32(file, "distances");
    let _ds_distance = dataset::normalize_all(ds_distance);

    let _ds_neighbors = dataset::get_dataset_i32(file, "neighbors");


    let time_start = Instant::now();

    // linear scan
    println!("bruteforce_search started at {:?}", time_start);
    bruteforce_search(_ds_test, _ds_train);

    let time_finish = Instant::now();
    println!("Duration: {:?}", time_finish.duration_since(time_start));
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn another() {
        panic!("Make this test fail");
    }
}