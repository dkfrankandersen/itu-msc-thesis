use ndarray::{Array,ArrayView2, s};
use rand::{prelude::*};
pub use ordered_float::*;
use crate::util::{sampling::sampling_without_replacement};
use crate::algs::{distance::cosine_similarity, pq_common::{Centroid}};

pub fn pq_kmeans<T: RngCore>(rng: T, k_centroids: usize, max_iterations: usize, dataset: &ArrayView2::<f64>, verbose_print: bool) -> Vec::<Centroid> {
        
    let datapoint_dimension = dataset.ncols();

    // Init
    let mut centroids = Vec::<Centroid>::with_capacity(k_centroids);
    let unique_indexes = sampling_without_replacement(rng, dataset.nrows(), k_centroids);

    // let dist_uniform = Uniform::new(0, dataset.nrows());
    for k in 0..k_centroids {
        // let rand_index = rng.sample(dist_uniform);
        let datapoint = dataset.slice(s![unique_indexes[k],..]);
        centroids.push(Centroid{id: k, point: datapoint.to_owned(), indexes: Vec::<usize>::new()});
    }

    // Repeat
    let mut last_centroids = Vec::<Centroid>::with_capacity(k_centroids);
    for iterations in 0..max_iterations  {
        if centroids == last_centroids {
            if verbose_print { println!("Computation has converged, iterations: {}", iterations); }
            break;
        }

        last_centroids = centroids.clone();

        // Remove centroid children
        centroids.iter_mut().for_each(|c| c.indexes.clear());
        
        // Assign
        for (index, datapoint) in dataset.outer_iter().enumerate() {
            let mut best_match: (f64, usize) = (f64::NEG_INFINITY, 0);
            for centroid in centroids.iter() {
                let distance = cosine_similarity(&centroid.point.view() , &datapoint);
                if OrderedFloat(best_match.0) < OrderedFloat(distance) { 
                    best_match = (distance, centroid.id); 
                }
            }
            centroids[best_match.1].indexes.push(index);
        }
        
        // Update
        for centroid in centroids.iter_mut() {
            if centroid.indexes.len() > 0 {

                // Clear centroid point
                centroid.point = Array::from_elem(datapoint_dimension, 0.);
                
                // Add dimension value of each
                for index in centroid.indexes.iter() {
                    let point = dataset.slice(s![*index,..]);
                    for (i, x) in point.iter().enumerate() {
                        centroid.point[i] += x;
                    }
                }

                // Divide by indexes to get mean
                let centroid_indexes_count = centroid.indexes.len() as f64;
                for i in 0..datapoint_dimension {  
                    centroid.point[i] = centroid.point[i]/centroid_indexes_count;
                }
            }
        }
    }
    centroids
}


#[cfg(test)]
mod pq_kmeans_tests {
    use ndarray::{Array2, arr2};
    use crate::algs::pq_kmeans::{pq_kmeans};

    fn dataset1() -> Array2<f64> {
        arr2(&[
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
            [3.0, 3.1, 3.2, 3.3, 3.4, 3.5],
            [4.0, 4.1, 4.2, 4.3, 4.4, 4.5],
            [5.0, 5.1, 5.2, 5.3, 5.4, 5.5],
            [6.0, 6.1, 6.2, 6.3, 6.4, 6.5],
            [7.0, 7.1, 7.2, 7.3, 7.4, 7.5],
            [8.0, 8.1, 8.2, 8.3, 8.4, 8.5],
            [9.0, 9.1, 9.2, 9.3, 9.4, 9.5],
        ])
    }
    #[test]
    fn kmeans_with_k_7_clusters_7() {
        use rand::{SeedableRng, rngs::StdRng};
        let rng = StdRng::seed_from_u64(11);
        let centroids = pq_kmeans(rng, 7, 200, &dataset1().view(), false);
        println!("{:?}", centroids);
        assert!(centroids.len() == 7);
    }

    #[test]
    fn kmeans_with_k_3_seed_111_centroid0_is() {
        use rand::{SeedableRng, rngs::StdRng};
        let rng = StdRng::seed_from_u64(111);
        let centroids = pq_kmeans(rng, 4, 200, &dataset1().view(), false);
        assert!(centroids[0].indexes == vec![3, 4, 5]);
    }
}