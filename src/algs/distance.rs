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

// pub fn distance_from_metric(p: &ArrayView1::<f64>, q: &ArrayView1::<f64>, metric: &str) -> f64 {
//     return cosine_similarity(p, q);
//     // if metric == "euclidian" {
//     //     return euclidian(p, q);
//     // } else if metric == "angular" {
//     //     return p.dot(q);
//     //     // return angular_similarity(p, q);
//     // } else if metric == "cosine" {
//     //     return cosine_similarity(p, q);
//     // } else if metric == "dot_product" {
//     //     return p.dot(q);
//     // } else {
//     //     panic!("Wrong or missing metric");
//     // }
// }

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

        assert!(expect_f64_near!(distance, 1.4142).is_ok());
    }
    #[test]
    fn given_2d_origin_to_origin() {
        let p: Array1::<f64> = arr1(&[0.0, 0.0]);
        let q: Array1::<f64> = arr1(&[0.0, 0.0]);
        let distance = euclidian(&p.view(), &q.view());

        assert!(expect_f64_near!(distance, 0.0).is_ok());
    }
    #[test]
    fn given_2d_origin_to_neg_point() {
        let p: Array1::<f64> = arr1(&[0.0, 0.0]);
        let q: Array1::<f64> = arr1(&[-2.0, -1.0]);
        let distance = euclidian(&p.view(), &q.view());
        println!("{}", distance);
        assert!(expect_f64_near!(distance, 2.23606797749979).is_ok());
    }
}

#[cfg(test)]
mod angular_similarity_tests {
    use ndarray::{Array1, arr1};
    use crate::algs::distance::{angular_similarity};
    use assert_float_eq::*;

    #[test]
    fn given_2d_origin_to_point() {
        let p: Array1::<f64> = arr1(&[0.0, 0.0]);
        let q: Array1::<f64> = arr1(&[1.0, 1.0]);
        let distance = angular_similarity(&p.view(), &q.view());

        let _assert = expect_float_relative_eq!(distance, 1.4142, 0.0001);
    }
    #[test]
    fn given_2d_origin_to_origin() {
        let p: Array1::<f64> = arr1(&[0.0, 0.0]);
        let q: Array1::<f64> = arr1(&[0.0, 0.0]);
        let distance = angular_similarity(&p.view(), &q.view());

        let _assert = expect_float_absolute_eq!(distance, 0.0, 0.0);
    }
    #[test]
    fn given_2d_origin_to_neg_point() {
        let p: Array1::<f64> = arr1(&[0.0, 0.0]);
        let q: Array1::<f64> = arr1(&[-2.0, -1.0]);
        let distance = angular_similarity(&p.view(), &q.view());

        let _assert = expect_float_relative_eq!(distance, 2.2360, 0.0001);
    }
}

#[cfg(test)]
mod cosine_similarity_tests {
    use ndarray::{Array1, arr1};
    use crate::algs::distance::{cosine_similarity};
    use assert_float_eq::*;

    #[test]
    fn given_2d_origin_to_point() {
        let p: Array1::<f64> = arr1(&[0.0, 0.0]);
        let q: Array1::<f64> = arr1(&[1.0, 1.0]);
        let distance = cosine_similarity(&p.view(), &q.view());

        let _assert = expect_float_relative_eq!(distance, 1.4142, 0.0001);
    }
    #[test]
    fn given_2d_origin_to_origin() {
        let p: Array1::<f64> = arr1(&[0.0, 0.0]);
        let q: Array1::<f64> = arr1(&[0.0, 0.0]);
        let distance = cosine_similarity(&p.view(), &q.view());

        let _assert = expect_float_absolute_eq!(distance, 0.0, 0.0);
    }
    #[test]
    fn given_2d_origin_to_neg_point() {
        let p: Array1::<f64> = arr1(&[0.0, 0.0]);
        let q: Array1::<f64> = arr1(&[-2.0, -1.0]);
        let distance = cosine_similarity(&p.view(), &q.view());

        let _assert = expect_float_relative_eq!(distance, 2.2360, 0.0001);
    }
}