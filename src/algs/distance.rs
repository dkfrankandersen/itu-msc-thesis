use ndarray::{ArrayView1, ArrayView2, Array1, arr1, arr2};
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



#[derive(Debug, Clone)]
pub struct CosineSimilarity {
    dot_products: Vec::<f64>,
    query: Option<Array1<f64>>,
    query_dot_product: f64,
}

impl CosineSimilarity {
    pub fn new(dataset: &ArrayView2::<f64>) -> Self {
        CosineSimilarity {
            dot_products: dataset.outer_iter()
                                .map(|p| p.dot(&p))
                                .collect(),
            query: None,
            query_dot_product: 0f64,
        }
    }

    pub fn set_query_dot_product(&mut self, p: Array1::<f64>) {
        self.query = Some(p.clone());
        self.query_dot_product = p.dot(&p);
    }

    pub fn lookup_dot_product(&self, index: usize) -> f64 {
        self.dot_products[index]
    }

    pub fn given_magnitude(&self, index: usize, p: &ArrayView1::<f64>) -> f64 {
        let magnitude = self.dot_products[index]*self.query_dot_product;
        let q = self.query.as_ref().unwrap();
        let dot_prod = p.dot(q);
        let cos_sim = dot_prod / (magnitude);
        return cos_sim;
    }
}

// impl DistanceImpl for CosineSimilarity {
//     fn min_dist(&self, p: &ArrayView1::<f64>, q: &ArrayView1::<f64>) -> f64 {
//         let dot_prod = p.dot(q);
//         let magnitude_p = p.dot(p).sqrt();
//         let magnitude_q = q.dot(q).sqrt();
//         let cos_sim = dot_prod / (magnitude_p*magnitude_q);
//         cos_sim
//     }
    
//     fn name(&self) -> String {
//         "CosineSimilarity".to_string()
//     }
// }

// #[derive(Debug, Clone)]
// pub enum Distance {
//     CosineSimilarity(CosineSimilarity),
// }

// trait DistanceImpl {
//     fn min_dist(&self, p: &ArrayView1::<f64>, q: &ArrayView1::<f64>) -> f64;
//     fn name(&self) -> String;
// }

// impl DistanceImpl for Distance {
//     fn min_dist(&self, p: &ArrayView1::<f64>, q: &ArrayView1::<f64>) -> f64 {
//         match self {
//             Distance::CosineSimilarity(x) => x.min_dist(p, q)
//         }
//     }
//     fn name(&self) -> String {
//         match self {
//             Distance::CosineSimilarity(x) => x.name()
//         }
//     }
// }



// pub struct DistanceFactory {}

// impl DistanceFactory {
//     pub fn get(dist_type: DistanceMetric) -> Distance {
//         let d = CosineSimilarity::new();
//         let p = &arr1(&[]);
//         let q = &arr1(&[]);
//         let dv = d.min_dist(&p.view(), &q.view());
//         match dist_type {
//             DistanceMetric::CosineSimilarity => {
//                             Distance::CosineSimilarity(CosineSimilarity::new())
//                             },
//             _ => unimplemented!(),
//         }
//     }
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