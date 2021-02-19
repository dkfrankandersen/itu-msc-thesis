extern crate ndarray;
extern crate hdf5;
use std::str::FromStr;
use hdf5::types::VarLenUnicode;
use std::time::{Instant, Duration};
use ndarray::{ArrayView1, ArrayView2, s};
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


    let attrs = util::Attrs {
        algo: VarLenUnicode::from_str("bruteforce").unwrap(),
        batch_mode: false,
        best_search_time: 0.01,
        build_time: 0.02,
        candidates: 10.0,
        count: 10,
        dataset: VarLenUnicode::from_str("glove-100-angular").unwrap(),
        distance: VarLenUnicode::from_str("cosine").unwrap(),
        expect_extra: false,
        index_size: 0.03,
        name: VarLenUnicode::from_str("bruteforce(n_trees=100,search_k=100)").unwrap(),
        run_count: 3
    };

    // println!("{:?}", attrs);

    util::store_results(result, attrs);
}

