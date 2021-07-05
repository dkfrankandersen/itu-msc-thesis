
// use ndarray::prelude::*;
// // use ndarray_linalg::*;

// // fn r_parallel_residual_error(x: &ArrayView1::<f64>, q: &ArrayView1::<f64>) -> Array1::<f64> {
// //     // Takes dot product of the residuals (x-q) and x, then multiplie onto x and divides with the norm of x to the power of 2 (so just dot product).
// //     ((x-q).dot(x)*x) / x.dot(x) //.sqrt().powi(2)
// // }

// // fn r_orthogonal_residual_error(x: &ArrayView1::<f64>, q: &ArrayView1::<f64>) -> Array1::<f64> {
// //     (x-q) - r_parallel_residual_error(x, q)
// // }

// fn eta_value(threshold_t: f64, dimension: usize) -> f64 {
//     threshold_t.powi(2) / (1.-threshold_t.powi(2)) * ((dimension-1) as f64)
// }

// #[cfg(test)]
// mod fa_scann_util_tests {
//     use crate::algs::fa_scann_util::*;
//     use assert_float_eq::*;

//     #[test]
//     fn eta_value_with_t_0_2_and_d_100_return_4_125() {
//         let eta = eta_value(0.2, 100);
//         println!("{:?}", eta);
//         assert!(expect_f64_near!(eta, 4.125 as f64).is_ok());
//     }

//     #[test]
//     fn eta_value_with_t_0_and_d_100_return_0() {
//         let eta = eta_value(0.0, 100);
//         println!("{:?}", eta);
//         assert!(expect_f64_near!(eta, 0. as f64).is_ok());
//     }

//     #[test]
//     fn eta_value_with_t_1_and_d_100_return_4_125() {
//         let eta = eta_value(1.0, 100);
//         println!("{:?}", eta);
//         assert!(expect_f64_near!(eta, f64::INFINITY).is_ok());
//     }
// }

// pub fn compute_parallel_cost_multiplier(t: f64, squared_l2_norm: f64, dim: usize) -> f64 {
//     // ScaNN Paper Theorem 3.4
//     let parallel_cost: f64 = t.sqrt() / squared_l2_norm;
//     let perpendicular_cost: f64 = (1.0 - t.sqrt()) / squared_l2_norm / (dim - 1) as f64;

//     let result = parallel_cost / perpendicular_cost;
//     result
// }
