extern crate ndarray;
// extern crate ndarray_linalg;
use std::time::{Instant};
use ndarray::{Array2, ArrayView1, ArrayBase, ViewRepr, Dim, OwnedRepr};
use priority_queue::PriorityQueue;

fn dist_euclidian(p: &ArrayView1::<f32>, q: &ArrayView1::<f32>) -> f32 {
    let mut sum_val = 0.0;
    for i in 0..p.len() {
        sum_val += (p[i]-q[i]).powi(2);
    }
    return sum_val.sqrt();
}

fn dist_cosine_similarity(p: &ArrayView1::<f32>, q: &ArrayView1::<f32>) -> f32 {
    let dot_prod = p.dot(q);
    let magnitude_p = p.dot(p).sqrt();
    let magnitude_q = q.dot(q).sqrt();
    let cos_sim = dot_prod / (magnitude_p*magnitude_q);
    return cos_sim;
}

fn dist_angular_similarity(p: &ArrayView1::<f32>, q: &ArrayView1::<f32>) -> f32 {
    let dot_prod = p.dot(q);
    let magnitude_p = p.dot(p).sqrt();
    let magnitude_q = q.dot(q).sqrt();
    let cos_sim = dot_prod / (magnitude_p*magnitude_q);
    return cos_sim.acos() / std::f32::consts::PI;
}

fn get_dataset_f32(file: &hdf5::File, dataset:&str) -> Array2::<f32> {
    return (file).dataset(dataset).unwrap().read_2d::<f32>().unwrap();
}

fn get_dataset_i32(file: &hdf5::File, dataset:&str) -> Array2::<f32> {
    return (file).dataset(dataset).unwrap().read_2d::<f32>().unwrap();
}

fn normalize(p: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>) 
                        -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>> {
    let magnitude = p.dot(p).sqrt();
    return p.map(|e| e/magnitude);
}

fn bruteforce_search(ds_test: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>,
                        ds_train: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>) {
                           
        for (idx_test, test_vector) in ds_test.outer_iter().enumerate() {
            let mut best_dist_euc:f32 = f32::INFINITY;
            let mut best_dist_cos:f32 = f32::NEG_INFINITY;
            let mut best_dist_ang:f32 = f32::INFINITY;
            let mut best_index_euc:usize = 0;
            let mut best_index_cos:usize = 0;
            let mut best_index_ang:usize = 0;
            for (idx_train, train_vector) in ds_train.outer_iter().enumerate() {
                // let dist_euc = dist_euclidian(&test_vector, &row);
                // let dist_cos = dist_cosine_similarity(&test_vector, &row);
                // let dist_ang = dist_angular_similarity(&test_vector, &row);

                let test_vector_norm = normalize(&test_vector);
                let row_vector_norm = normalize(&train_vector);

                let dist_euc = dist_euclidian(&test_vector_norm.view(), &row_vector_norm.view());
                let dist_cos = dist_cosine_similarity(&test_vector_norm.view(), &row_vector_norm.view());
                let dist_ang = dist_angular_similarity(&test_vector_norm.view(), &row_vector_norm.view());
        
                if dist_euc < best_dist_euc {
                    best_index_euc = idx_train;
                    best_dist_euc = dist_euc;
                }

                if dist_cos > best_dist_cos {
                    best_index_cos = idx_train;
                    best_dist_cos = dist_cos;
                }

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
    let filename = "../../datasets/glove-100-angular.hdf5";
    let file = &hdf5::File::open(filename).unwrap();
    let ds_train = get_dataset_f32(file, "train");
    let ds_test = get_dataset_f32(file, "test");
    let _ds_distance = get_dataset_f32(file, "distances");
    let _ds_neighbors = get_dataset_i32(file, "neighbors");
    
    let time_start = Instant::now();

    // linear scan
    println!("bruteforce_search started at {:?}", time_start);
    bruteforce_search(&ds_test.view(), &ds_train.view());

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