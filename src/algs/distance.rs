use ndarray::{ArrayView1};
#[allow(dead_code)]
pub enum DistType {
    Angular,
    Cosine,
    Euclidian
}

#[allow(dead_code)]
pub fn euclidian(p: &ArrayView1::<f64>, q: &ArrayView1::<f64>) -> f64 {
    let mut sum_val = 0.0;
    for i in 0..p.len() {
        sum_val += (p[i]-q[i]).powi(2);
    }
    return sum_val.sqrt();
}

#[allow(dead_code)]
pub fn angular_similarity(p: &ArrayView1::<f64>, q: &ArrayView1::<f64>) -> f64 {
    let dot_prod = p.dot(q);
    let magnitude_p = p.dot(p).sqrt();
    let magnitude_q = q.dot(q).sqrt();
    let cos_sim = dot_prod / (magnitude_p*magnitude_q);
    return cos_sim.acos() / std::f64::consts::PI;
}


pub fn cosine_similarity(p: &ArrayView1::<f64>, q: &ArrayView1::<f64>) -> f64 {
    let dot_prod = p.dot(q);
    let magnitude_p = p.dot(p).sqrt();
    let magnitude_q = q.dot(q).sqrt();
    let cos_sim = dot_prod / (magnitude_p*magnitude_q);
    return cos_sim;
}

// double ComputeParallelCostMultiplier(double t, double squared_l2_norm, DimensionIndex dims) {
//     const double parallel_cost = Square(t) / squared_l2_norm;
//     const double perpendicular_cost =
//     (1.0 - Square(t) / squared_l2_norm) / (dims - 1.0);
//     return parallel_cost / perpendicular_cost;
// }

pub fn _compute_parallel_cost_multiplier(t: f64, squared_l2_norm: f64) -> f64 {
    // ScaNN Paper Theorem 3.4
    let parallel_cost: f64 = t.sqrt() / squared_l2_norm;
    let perpendicular_cost: f64 = (1.0 - t.sqrt()) / squared_l2_norm;

    let result = parallel_cost / perpendicular_cost;
    result

}

pub fn _anisotropic_loss() {

}

#[cfg(test)]
mod euclidian_tests {
    use ndarray::{Array1, arr1};
    use crate::algs::distance::{euclidian};
    use assert_float_eq::*;

    #[test]
    fn given_2d_origin_to_point() {
        let p: Array1::<f64> = arr1(&[0.0, 0.0]);
        let q: Array1::<f64> = arr1(&[1.0, 1.0]);
        let distance = euclidian(&p.view(), &q.view());

        let _assert = expect_float_relative_eq!(distance, 1.4142, 0.0001);
    }
    #[test]
    fn given_2d_origin_to_origin() {
        let p: Array1::<f64> = arr1(&[0.0, 0.0]);
        let q: Array1::<f64> = arr1(&[0.0, 0.0]);
        let distance = euclidian(&p.view(), &q.view());

        let _assert = expect_float_absolute_eq!(distance, 0.0, 0.0);
    }
    #[test]
    fn given_2d_origin_to_neg_point() {
        let p: Array1::<f64> = arr1(&[0.0, 0.0]);
        let q: Array1::<f64> = arr1(&[-2.0, -1.0]);
        let distance = euclidian(&p.view(), &q.view());

        let _assert = expect_float_relative_eq!(distance, 2.2360, 0.0001);
    }
}