extern crate ndarray;
extern crate hdf5;
use std::time::{Instant, Duration};
use ndarray::{ArrayView1, ArrayView2, s};
mod algs;
use algs::dataset::Dataset;

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
}

