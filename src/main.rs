extern crate ndarray;
extern crate hdf5;
use std::time::{Instant};
mod algs;
use algs::dataset::Dataset;
use algs::bruteforce;

fn main() {
    let filename = "datasets/glove-100-angular.hdf5";
    let ds = Dataset::new(filename);
    let ds_train_norm = ds.train_normalize();
    let ds_test_norm = ds.test_normalize();
    let ds_distances_norm = ds.distances_normalize();
    let ds_neighbors = ds.neighbors();

    let time_start = Instant::now();

    println!("start ds_neighbor for 0 and dist to 5 closests: ");
    for (i, v) in ds_neighbors.outer_iter().enumerate() {
        println!("{} {:?} {}", i, v[0], ds_distances_norm[[i,0]]);
        println!("{} {:?} {}", i, v[1], ds_distances_norm[[i,1]]);
        println!("{} {:?} {}", i, v[2], ds_distances_norm[[i,2]]);
        println!("{} {:?} {}", i, v[3], ds_distances_norm[[i,3]]);
        println!("{} {:?} {}", i, v[4], ds_distances_norm[[i,4]]);
        break
    }
    println!("end ds_neighbors: ");
    
    // bruteforce_search
    println!("bruteforce_search started at {:?}", time_start);
    for (i,p) in ds_test_norm.outer_iter().enumerate() {
        println!("\n-- Test index: {:?} --", i);
        let res1 = bruteforce::single_search(&p, &ds_train_norm.view(), 10);
        println!("{:?}", res1);
        break;
    }

    let time_finish = Instant::now();
    println!("Duration: {:?}", time_finish.duration_since(time_start));
}