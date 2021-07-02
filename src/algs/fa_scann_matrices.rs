// extern crate approx; // For the macro relative_eq!
// extern crate nalgebra as na;
// use na::{Vector3, Rotation3, Matrix2x3, Vector2, Matrix4, LU, DMatrix};
// use ndarray::{Array, Array2, arr3};

// pub fn matrix_from_index_and_dataset(indexes: Vec<usize>, dataset: Array2::<f64>) {

//     let n = indexes.len();
//     let m = dataset.cols();
//     let mut matrix = DMatrix::from_element(4, 4, 0.0);
//     matrix[(0, 0)] = 1.;

//     println!("{:?}", matrix);
    
// }
// #[cfg(test)]
// mod fa_scann_matrices_tests {
//     use crate::algs::fa_scann_matrices::*;
//     use assert_float_eq::*;

//     #[test]
//     fn matrix_from_index_and_dataset_test() {
//         let indexes = vec![1, 2, 3];
//         let dataset = Array::from_elem((5, 100), 0.);
//         matrix_from_index_and_dataset(indexes, dataset);
//     }
// }