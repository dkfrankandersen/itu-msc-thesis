extern crate ndarray;
extern crate hdf5;
use std::time::{Instant};
mod algs;
use algs::{dataset,linearsearch};


fn main() {
    let _e = hdf5::silence_errors();
    let _filename = "datasets/glove-100-angular.hdf5";
    let file = &dataset::get_dataset(_filename);
    
    let ds_train = dataset::get_dataset_f64(file, "train");
    let ds_train_norm = dataset::normalize_all(ds_train);
    
    let ds_test = dataset::get_dataset_f64(file, "test");
    let ds_test_norm = &dataset::normalize_all(ds_test);

    let ds_distance = dataset::get_dataset_f64(file, "distances");
    let ds_distance_norm = dataset::normalize_all(ds_distance);

    let ds_neighbors = dataset::get_dataset_i64(file, "neighbors");

    let time_start = Instant::now();

    // linear scan
    println!("bruteforce_search started at {:?}", time_start);
    for (i,p) in ds_test_norm.outer_iter().enumerate() {
        let res = linearsearch::single_search(&p, &ds_train_norm.view(), 1000);
        println!("{:?}", res);
        println!("{:?}", ds_neighbors[[i,0]]);
        break;
    }


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