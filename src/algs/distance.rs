use ndarray::{ArrayView1, ArrayView2};
use ordered_float::OrderedFloat;
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum DistanceMetric {
    Angular,
    CosineSimilarity,
    Euclidian,
    DotProduct
}

pub fn min_distance(a: &ArrayView1::<f64>, b: &ArrayView1::<f64>, metric: &DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::DotProduct => -(a.dot(b)),
        DistanceMetric::Euclidian => euclidian(a, b),
        DistanceMetric::Angular => angular_similarity(a, b),
        DistanceMetric::CosineSimilarity => cosine_similarity(a, b),
        _ => panic!("DistanceMetric unknown")
    }
    
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

// #[derive(Debug, Clone)]
// pub struct CosineSimilarity {
//     dot_products_sqrt: Vec::<f64>,
// }

// impl CosineSimilarity {
//     pub fn new(dataset: &ArrayView2::<f64>) -> Self {
//         CosineSimilarity {
//             dot_products_sqrt: dataset.outer_iter()
//             .map(|p| p.dot(&p).sqrt())
//             .collect()
//         }
//     }

//     pub fn cosine_similarity(&self, p: &ArrayView1::<f64>, q: &ArrayView1::<f64>) -> f64 {
//         let dot_prod = p.dot(q);
//         let magnitude_p = p.dot(p).sqrt();
//         let magnitude_q = q.dot(q).sqrt();
//         dot_prod / (magnitude_p*magnitude_q)
//     }

//     pub fn query_dot_sqrt(&self, q: &ArrayView1::<f64>) -> f64 {
//         q.dot(q).sqrt()
//     }

//     pub fn distance(&self, p: &ArrayView1::<f64>, q: &ArrayView1::<f64>) -> f64 {
//         self.cosine_similarity(p, q)
//     }

//     pub fn min_distance_ordered(&self, p: &ArrayView1::<f64>, q: &ArrayView1::<f64>) -> OrderedFloat::<f64> {
//         OrderedFloat(-self.distance(p, q))
//     }

//     pub fn max_distance_ordered(&self, p: &ArrayView1::<f64>, q: &ArrayView1::<f64>) -> OrderedFloat::<f64> {
//         OrderedFloat(self.distance(p, q))
//     }

//     pub fn fast_cosine_similarity(&self, p_index: usize, p: &ArrayView1::<f64>, q: &ArrayView1::<f64>, q_dot_sqrt: f64) -> f64 {
//         let magnitude = self.dot_products_sqrt[p_index]*q_dot_sqrt;
//         let dot_prod = p.dot(q);
//         dot_prod / (magnitude)
//     }

//     pub fn fast_distance(&self, p_index: usize, p: &ArrayView1::<f64>, q: &ArrayView1::<f64>, q_dot_sqrt: f64) -> f64 {
//         self.fast_cosine_similarity(p_index, p, q, q_dot_sqrt)
//     }

//     pub fn fast_min_distance_ordered(&self, p_index: usize, p: &ArrayView1::<f64>, q: &ArrayView1::<f64>, q_dot_sqrt: f64) -> OrderedFloat::<f64> {
//         OrderedFloat(-self.fast_distance(p_index, p, q, q_dot_sqrt))
//     }

//     pub fn fast_max_distance_ordered(&self, p_index: usize, p: &ArrayView1::<f64>, q: &ArrayView1::<f64>, q_dot_sqrt: f64) -> OrderedFloat::<f64> {
//         OrderedFloat(self.fast_distance(p_index, p, q, q_dot_sqrt))
//     }
// }