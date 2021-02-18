extern crate ndarray;
extern crate hdf5;
use std::time::{Instant};
mod algs;
use algs::{dataset,pq,bruteforce};
use algs::dataset::Dataset;

fn main() {
    let _e = hdf5::silence_errors();
    let _filename = "datasets/glove-100-angular.hdf5";
    // let file = &dataset::get_dataset(_filename);

    let ds = Dataset::new(_filename);

    let ds_train_norm = ds.get_dataset_normalize("train");
    let ds_test_norm = ds.get_dataset_normalize("test");
    let ds_distance_norm = ds.get_dataset_normalize("distances");
    let ds_neighbors = ds.get_dataset_usize("neighbors");
    
    // let ds_train = dataset::get_dataset_f64(file, "train");
    // let ds_train_norm = dataset::normalize_all(ds_train);
    
    // let ds_test = dataset::get_dataset_f64(file, "test");
    // let ds_test_norm = &dataset::normalize_all(ds_test);

    // let ds_distance = dataset::get_dataset_f64(file, "distances");
    // let ds_distance_norm = dataset::normalize_all(ds_distance);

    // let ds_neighbors = dataset::get_dataset_usize(file, "neighbors");

    let time_start = Instant::now();

    println!("start ds_neighbor for 0 and dist to 5 closests: ");
    for (i, v) in ds_neighbors.outer_iter().enumerate() {
        println!("{} {:?} {}", i, v[0], ds_distance_norm[[i,0]]);
        println!("{} {:?} {}", i, v[1], ds_distance_norm[[i,1]]);
        println!("{} {:?} {}", i, v[2], ds_distance_norm[[i,2]]);
        println!("{} {:?} {}", i, v[3], ds_distance_norm[[i,3]]);
        println!("{} {:?} {}", i, v[4], ds_distance_norm[[i,4]]);
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