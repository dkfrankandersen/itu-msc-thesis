extern crate ndarray;
extern crate hdf5;
use std::time::{Instant, Duration};
use ndarray::{ArrayView1, ArrayView2, s};
mod algs;
use algs::dataset::Dataset;
use algs::bruteforce;

fn print_true_neighbors(ds: ArrayView2<usize>, from : usize, to: usize, m: usize) {
    println!("| Distance for 5 closests neighbors from {} to {}:", from, to);
    for i in from..to {
        let mut neighbors = Vec::new();
        for j in 0..m {
            neighbors.push(ds[[i,j]]);
        }
        println!("|  idx: {} neighbors {:?}", i, neighbors);
    }
    println!("");
}

fn single_query(p: &ArrayView1<f64>, dataset: &ArrayView2<f64>) -> (Duration, Vec<usize>) {
    let time_start = Instant::now();
    let candidates = bruteforce::query(&p, &dataset, 10);
    let time_finish = Instant::now();
    let total_time = time_finish.duration_since(time_start);
    (total_time, candidates)
}

fn main() {
    let filename = "datasets/glove-100-angular.hdf5";
    let ds = Dataset::new(filename);
    let ds_train_norm = ds.train_normalize();
    let ds_test_norm = ds.test_normalize();
    let ds_distances_norm = ds.distances_normalize();
    let ds_neighbors = ds.neighbors();

    print_true_neighbors(ds_neighbors.view(), 0, 5, 10);

    let v = &ds_test_norm.slice(s![0,..]);
    let result = single_query(v, &ds_train_norm.view());
    println!("{:?}", result);

}

