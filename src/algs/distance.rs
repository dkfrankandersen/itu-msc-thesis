use ndarray::{ArrayView1, ArrayView2};
use ordered_float::OrderedFloat;
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum DistanceMetric {
    Angular,
    CosineSimilarity,
    Euclidian
}

#[allow(dead_code)]
pub fn euclidian(p: &ArrayView1::<f64>, q: &ArrayView1::<f64>) -> f64 {
    let mut sum_val = 0.0;
    for i in 0..p.len() {
        sum_val += (p[i]-q[i]).powi(2);
    }
    sum_val.sqrt()
}

#[allow(dead_code)]
pub fn angular_similarity(p: &ArrayView1::<f64>, q: &ArrayView1::<f64>) -> f64 {
    let dot_prod = p.dot(q);
    let magnitude_p = p.dot(p).sqrt();
    let magnitude_q = q.dot(q).sqrt();
    let cos_sim = dot_prod / (magnitude_p*magnitude_q);
    cos_sim.acos() / std::f64::consts::PI
}

pub fn cosine_similarity(p: &ArrayView1::<f64>, q: &ArrayView1::<f64>) -> f64 {
    let dot_prod = p.dot(q);
    let magnitude_p = p.dot(p).sqrt();
    let magnitude_q = q.dot(q).sqrt();
    dot_prod / (magnitude_p*magnitude_q)
}

#[derive(Debug, Clone)]
pub struct CosineSimilarity {
    dot_products_sqrt: Vec::<f64>,
}

impl CosineSimilarity {
    pub fn new(dataset: &ArrayView2::<f64>) -> Self {
        CosineSimilarity {
            dot_products_sqrt: dataset.outer_iter()
            .map(|p| p.dot(&p).sqrt())
            .collect()
        }
    }

    pub fn cosine_similarity(&self, p: &ArrayView1::<f64>, q: &ArrayView1::<f64>) -> f64 {
        let dot_prod = p.dot(q);
        let magnitude_p = p.dot(p).sqrt();
        let magnitude_q = q.dot(q).sqrt();
        dot_prod / (magnitude_p*magnitude_q)
    }

    pub fn query_dot_sqrt(&self, q: &ArrayView1::<f64>) -> f64 {
        q.dot(q).sqrt()
    }

    pub fn distance(&self, p: &ArrayView1::<f64>, q: &ArrayView1::<f64>) -> f64 {
        self.cosine_similarity(p, &q)
    }

    pub fn min_distance_ordered(&self, p: &ArrayView1::<f64>, q: &ArrayView1::<f64>) -> OrderedFloat::<f64> {
        OrderedFloat(-self.distance(p, q))
    }

    pub fn max_distance_ordered(&self, p: &ArrayView1::<f64>, q: &ArrayView1::<f64>) -> OrderedFloat::<f64> {
        OrderedFloat(self.distance(p, q))
    }

    pub fn fast_cosine_similarity(&self, p_index: usize, p: &ArrayView1::<f64>, q: &ArrayView1::<f64>, q_dot_sqrt: f64) -> f64 {
        let magnitude = self.dot_products_sqrt[p_index]*q_dot_sqrt;
        let dot_prod = p.dot(q);
        dot_prod / (magnitude)
    }

    pub fn fast_distance(&self, p_index: usize, p: &ArrayView1::<f64>, q: &ArrayView1::<f64>, q_dot_sqrt: f64) -> f64 {
        self.fast_cosine_similarity(p_index, p, q, q_dot_sqrt)
    }

    pub fn fast_min_distance_ordered(&self, p_index: usize, p: &ArrayView1::<f64>, q: &ArrayView1::<f64>, q_dot_sqrt: f64) -> OrderedFloat::<f64> {
        OrderedFloat(-self.fast_distance(p_index, p, q, q_dot_sqrt))
    }

    pub fn fast_max_distance_ordered(&self, p_index: usize, p: &ArrayView1::<f64>, q: &ArrayView1::<f64>, q_dot_sqrt: f64) -> OrderedFloat::<f64> {
        OrderedFloat(self.fast_distance(p_index, p, q, q_dot_sqrt))
    }
}

// #[cfg(test)]
// mod euclidian_tests {
//     use ndarray::{Array1, arr1};
//     use crate::algs::distance::{euclidian};
//     use assert_float_eq::*;

//     #[test]
//     fn given_2d_origin_to_point() {
//         let p: Array1::<f64> = arr1(&[0.0, 0.0]);
//         let q: Array1::<f64> = arr1(&[1.0, 1.0]);
//         let distance = euclidian(&p.view(), &q.view());

//         assert!(expect_f64_near!(distance, 1.4142).is_ok());
//     }
//     #[test]
//     fn given_2d_origin_to_origin() {
//         let p: Array1::<f64> = arr1(&[0.0, 0.0]);
//         let q: Array1::<f64> = arr1(&[0.0, 0.0]);
//         let distance = euclidian(&p.view(), &q.view());

//         assert!(expect_f64_near!(distance, 0.0).is_ok());
//     }
//     #[test]
//     fn given_2d_origin_to_neg_point() {
//         let p: Array1::<f64> = arr1(&[0.0, 0.0]);
//         let q: Array1::<f64> = arr1(&[-2.0, -1.0]);
//         let distance = euclidian(&p.view(), &q.view());
//         println!("{}", distance);
//         assert!(expect_f64_near!(distance, 2.23606797749979).is_ok());
//     }
// }

// #[cfg(test)]
// mod angular_similarity_tests {
//     use ndarray::{Array1, arr1};
//     use crate::algs::distance::{angular_similarity};
//     use assert_float_eq::*;

//     #[test]
//     fn given_2d_origin_to_point() {
//         let p: Array1::<f64> = arr1(&[0.0, 0.0]);
//         let q: Array1::<f64> = arr1(&[1.0, 1.0]);
//         let distance = angular_similarity(&p.view(), &q.view());

//         let _assert = expect_float_relative_eq!(distance, 1.4142, 0.0001);
//     }
//     #[test]
//     fn given_2d_origin_to_origin() {
//         let p: Array1::<f64> = arr1(&[0.0, 0.0]);
//         let q: Array1::<f64> = arr1(&[0.0, 0.0]);
//         let distance = angular_similarity(&p.view(), &q.view());

//         let _assert = expect_float_absolute_eq!(distance, 0.0, 0.0);
//     }
//     #[test]
//     fn given_2d_origin_to_neg_point() {
//         let p: Array1::<f64> = arr1(&[0.0, 0.0]);
//         let q: Array1::<f64> = arr1(&[-2.0, -1.0]);
//         let distance = angular_similarity(&p.view(), &q.view());

//         let _assert = expect_float_relative_eq!(distance, 2.2360, 0.0001);
//     }
// }

// #[cfg(test)]
// mod cosine_similarity_tests {
//     use ndarray::{Array1, arr1};
//     use crate::algs::distance::{cosine_similarity};
//     use assert_float_eq::*;

//     #[test]
//     fn given_2d_origin_to_point() {
//         let p: Array1::<f64> = arr1(&[0.0, 0.0]);
//         let q: Array1::<f64> = arr1(&[1.0, 1.0]);
//         let distance = cosine_similarity(&p.view(), &q.view());

//         let _assert = expect_float_relative_eq!(distance, 1.4142, 0.0001);
//     }
//     #[test]
//     fn given_2d_origin_to_origin() {
//         let p: Array1::<f64> = arr1(&[0.0, 0.0]);
//         let q: Array1::<f64> = arr1(&[0.0, 0.0]);
//         let distance = cosine_similarity(&p.view(), &q.view());

//         let _assert = expect_float_absolute_eq!(distance, 0.0, 0.0);
//     }
//     #[test]
//     fn given_2d_origin_to_neg_point() {
//         let p: Array1::<f64> = arr1(&[0.0, 0.0]);
//         let q: Array1::<f64> = arr1(&[-2.0, -1.0]);
//         let distance = cosine_similarity(&p.view(), &q.view());

//         let _assert = expect_float_relative_eq!(distance, 2.2360, 0.0001);
//     }
// }