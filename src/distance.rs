use ndarray::{ArrayView1};

pub fn dist_euclidian(p: &ArrayView1::<f32>, q: &ArrayView1::<f32>) -> f32 {
    let mut sum_val = 0.0;
    for i in 0..p.len() {
        sum_val += (p[i]-q[i]).powi(2);
    }
    return sum_val.sqrt();
}

pub fn dist_cosine_similarity(p: &ArrayView1::<f32>, q: &ArrayView1::<f32>) -> f32 {
    let dot_prod = p.dot(q);
    let magnitude_p = p.dot(p).sqrt();
    let magnitude_q = q.dot(q).sqrt();
    let cos_sim = dot_prod / (magnitude_p*magnitude_q);
    return cos_sim;
}

pub fn dist_angular_similarity(p: &ArrayView1::<f32>, q: &ArrayView1::<f32>) -> f32 {
    let dot_prod = p.dot(q);
    let magnitude_p = p.dot(p).sqrt();
    let magnitude_q = q.dot(q).sqrt();
    let cos_sim = dot_prod / (magnitude_p*magnitude_q);
    return cos_sim.acos() / std::f32::consts::PI;
}

enum DistType {
    Angular,
    Cosine,
    Euclidian
}