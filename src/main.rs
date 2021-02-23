extern crate ndarray;
extern crate hdf5;
use std::time::{Instant, Duration};
use ndarray::{s};
mod algs;
use algs::dataset::Dataset;
mod util;

fn main() {
    let filename = "datasets/glove-100-angular.hdf5";
    let ds = Dataset::new(filename);
    let ds_train_norm = ds.train_normalize();
    let ds_test_norm = ds.test_normalize();
    let ds_distances_norm = ds.distances_normalize();
    let ds_neighbors = ds.neighbors();

    ds.print_true_neighbors(0, 5, 10);

    let v = &ds_test_norm.slice(s![0,..]);
    let result = algs::single_query(v, &ds_train_norm.view());
    println!("{:?}", result);
    
    let attrs = util::Attributes {
        algo: "bruteforce".to_string(),
        batch_mode: false,
        best_search_time: 0.01,
        build_time: 0.02,
        candidates: 10.0,
        count: 10,
        dataset: "glove-100-angular".to_string(),
        distance: "cosine".to_string(),
        expect_extra: false,
        index_size: 0.03,
        name: "bruteforce(n_trees=100,search_k=100)".to_string(),
        run_count: 3
    };

    // println!("{:?}", result);
    // println!("{:?}", attrs);

    let saved = util::store_results(result, attrs);
    println!("{:?}", saved);
}

