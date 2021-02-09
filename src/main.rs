#[macro_use]
extern crate ndarray;
use std::time::{Instant};


fn dist_euclidian(p: &ndarray::Array1::<f32>, q: &ndarray::Array1::<f32>) -> f32 {
    let mut sum_val = 0.0;
    for i in 0..p.len() {
        sum_val += (p[i]-q[i]).powi(2);
    }
    return sum_val.sqrt();
}

fn dist_cosine_similarity(p: &ndarray::Array1::<f32>, q: &ndarray::Array1::<f32>) -> f32 {
    let dot_prod = p.dot(q);
    let magnitude_p = p.dot(p).sqrt();
    let magnitude_q = p.dot(p).sqrt();
    let cos_sim = dot_prod / (magnitude_p*magnitude_q);
    return cos_sim;
}

fn dist_angular_similarity(p: &ndarray::Array1::<f32>, q: &ndarray::Array1::<f32>) -> f32 {
    let dot_prod = p.dot(q);
    let magnitude_p = p.dot(p).sqrt();
    let magnitude_q = p.dot(p).sqrt();
    let cos_sim = dot_prod / (magnitude_p*magnitude_q);
    return cos_sim.acos() / std::f32::consts::PI;
}

fn get_dataset_f32(file: &hdf5::File, dataset:&str) -> ndarray::Array2::<f32> {
    return (file).dataset(dataset).unwrap().read_2d::<f32>().unwrap();
}

fn get_dataset_i32(file:&hdf5::File, dataset:&str) -> ndarray::Array2::<f32> {
    return (file).dataset(dataset).unwrap().read_2d::<f32>().unwrap();
}

fn bruteforce_search(ds_test: &ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>,
                        ds_train: &ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>) {

        // let first = &ds_test.slice(s![0,..]);
        
        
        for i in 0..ds_test.len() {
            let mut best_dist:f32 = f32::INFINITY;
            let mut best_index:i32 = -1;
            let mut j:i32 = 0;
            let first = &ds_test.slice(s![i,..]);
            for row in ds_train.outer_iter() {
                // let dist = dist_euclidian(&first.to_owned(), &row.to_owned());
                // let dist = dist_cosine_similarity(&first.to_owned(), &row.to_owned());
                let dist = dist_angular_similarity(&first.to_owned(), &row.to_owned());
        
                if dist < best_dist {
                    best_dist = dist;
                    best_index = j;
                }
                j += 1;
            }
            println!("Best index {} with dist: {}", best_index, best_dist);
        }
}

fn main() {

    let filename = "../../datasets/glove-100-angular.hdf5";
    let file = &hdf5::File::open(filename).unwrap();
    let ds_train = get_dataset_f32(file, "train");
    let ds_test = get_dataset_f32(file, "test");
    let _ds_distance = get_dataset_f32(file, "distances");
    let _ds_neighbors = get_dataset_i32(file, "neighbors");
    

    let time_start = Instant::now();

    // linear scan
    println!("bruteforce_search started at {:?}", time_start);
    bruteforce_search(&ds_test, &ds_train);

    
    let time_finish = Instant::now();
    println!("Duration: {:?}", time_finish.duration_since(time_start));
    
}
