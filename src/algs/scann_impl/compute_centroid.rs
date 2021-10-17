use crate::algs::scann_impl::common::*;
use ndarray::prelude::*;
use ndarray_linalg::*;

#[allow(dead_code)]
pub fn recompute_centroids_simple(spherical: bool, dataset: &ArrayView2::<f64>, centroids: &mut Vec::<Centroid>) {
    // Update new means
    for centroid in centroids.iter_mut() {
        // Clear centroid point
        centroid.point.fill(0.);

        if centroid.indexes.len() > 0 {
            // Add dimension value of each
            for index in centroid.indexes.iter() {
                let point = dataset.slice(s![*index,..]);
                for (i, x) in point.iter().enumerate() {
                    centroid.point[i] += x;
                }
            }

            let divisor = if spherical { centroid.point.dot(&centroid.point).sqrt() }
                          else { centroid.indexes.len() as f64};
     
            if divisor == 0. {
                println!("recompute_centroids_simple, could not normalize centroid due to zero norm or empty partition.");
                continue;
            }
            let multiplier = 1.0 / divisor;
            // Divide by indexes to get mean
            for i in 0..centroid.point.len() {  
                centroid.point[i] *= multiplier;
            }
        }
    }
}