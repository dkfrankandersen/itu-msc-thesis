use ndarray::{ArrayView1, ArrayView2};
use ordered_float::OrderedFloat;
use rayon::prelude::*;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum DistanceMetric {
    Angular,
    CosineSimilarity,
    Euclidian,
    SquareEuclidian,
    DotProduct
}

pub fn min_distance(a: &ArrayView1::<f64>, b: &ArrayView1::<f64>, metric: &DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::DotProduct => -(a.dot(b)),
        DistanceMetric::Euclidian => euclidian(a, b),
        DistanceMetric::SquareEuclidian => square_euclidian(a, b),
        DistanceMetric::Angular => angular_similarity(a, b),
        DistanceMetric::CosineSimilarity => cosine_similarity(a, b),
        _ => panic!("DistanceMetric unknown")
    }
}

#[allow(dead_code)]
pub fn euclidian(a: &ArrayView1::<f64>, b: &ArrayView1::<f64>) -> f64 {
    let sum_val: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    sum_val.sqrt()
}

pub fn square_euclidian(a: &ArrayView1::<f64>, b: &ArrayView1::<f64>) -> f64 {
    let sum_val: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    sum_val
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