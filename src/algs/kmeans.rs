use ndarray::{Array,ArrayView2, s, parallel::prelude::*, Axis};
use rand::{prelude::*};
pub use ordered_float::*;
use crate::util::{sampling::sampling_without_replacement};
use crate::algs::{distance::cosine_similarity, common::{Centroid}};
use indicatif::ProgressBar;
use crate::util::{DebugTimer};
use std::collections::{BinaryHeap, HashMap};
use std::thread;
use std::sync::Arc;

pub fn kmeans<T: RngCore>(rng: T, k_centroids: usize, max_iterations: usize, dataset: &ArrayView2::<f64>, verbose_print: bool) -> Vec::<Centroid> {
        
    let datapoint_dimension = dataset.ncols();

    // Init
    let mut centroids = Vec::<Centroid>::with_capacity(k_centroids);
    let unique_indexes = sampling_without_replacement(rng, dataset.nrows(), k_centroids);

    println!("Started kmeans Init");
    let mut t = DebugTimer::start("kmeans init");
    let bar_unique_indexes = ProgressBar::new(unique_indexes.len() as u64);
    for (k, index) in unique_indexes.iter().enumerate() {
        let datapoint = dataset.slice(s![*index,..]);
        centroids.push(Centroid{id: k, point: datapoint.to_owned(), indexes: Vec::<usize>::new()});
        bar_unique_indexes.inc(1);
    }
    bar_unique_indexes.finish();
    t.stop();
    t.print_as_millis();

    // Repeat
    println!("Started kmeans Repeat");
    let mut t = DebugTimer::start("kmeans Repeat");
    let bar_max_iterations = ProgressBar::new(max_iterations as u64);
    let mut last_centroids = Vec::<Centroid>::with_capacity(k_centroids);
    let dataset_arc = Arc::new(dataset.to_owned());
    for iterations in 0..max_iterations  {
        if centroids == last_centroids {
            if verbose_print { println!("Computation has converged, iterations: {}", iterations); }
            break;
        }

        // let mut t = DebugTimer::start("kmeans clone");
        last_centroids = centroids.clone();
        // t.stop();
        // t.print_as_millis();

        // Remove centroid children
        // let mut t = DebugTimer::start("kmeans indexes clear");
        centroids.iter_mut().for_each(|c| c.indexes.clear());
        // t.stop();
        // t.print_as_nanos();

        // Assign
        // let mut t = DebugTimer::start("kmeans assign");
        // for (index, datapoint) in dataset.outer_iter().enumerate() {
        //     let mut best_distance: OrderedFloat::<f64> = OrderedFloat(f64::NEG_INFINITY);
        //     let mut best_index: usize = 0;
        //     for centroid in centroids.iter() {
        //         let distance = OrderedFloat(cosine_similarity(&centroid.point.view(), &datapoint));
        //         if distance > best_distance { 
        //             best_distance = distance;
        //             best_index = centroid.id; 
        //         }
        //     }
        //     centroids[best_index].indexes.push(index);
        // }

        let mut cen_points = Vec::new();
        for centroid in centroids.iter() {
            cen_points.push(centroid.point.to_owned());
        }

        let mut handles = Vec::new();
        const NTHREADS: usize = 4;
        let max_val = dataset.nrows();
        let chunk = max_val/NTHREADS;

        let mut chunks = Vec::<(usize,usize)>::new();
        for i in 0..NTHREADS {
            let from = chunk*i;
            let mut to = from+chunk;
            if max_val-to < chunk {
                to = max_val;
            }
            chunks.push((from, to));
        }
        let cen_points_arc = Arc::new(cen_points);
        for (f, t) in chunks.into_iter() {
            let points = Arc::clone(&cen_points_arc);
            let dataset_arc = Arc::clone(&dataset_arc);
            handles.push(thread::spawn(move || {
                let mut hmap = HashMap::<usize, Vec::<usize>>::new();
                for index in f..t {
                    let mut best_distance: OrderedFloat::<f64> = OrderedFloat(f64::NEG_INFINITY);
                    let mut best_index: usize = 0;
                    for (centroid_index, point) in points.iter().enumerate() {
                        let distance = OrderedFloat(cosine_similarity(&point.view(), &dataset_arc.slice(s![index,..])));
                        if distance > best_distance { 
                            best_distance = distance;
                            best_index = centroid_index; 
                        }
                    }
                    if hmap.get(&best_index).is_some() {
                        hmap.get_mut(&best_index).unwrap().push(index);
                    } else {
                        hmap.insert(best_index, vec!(index));
                    }
                }
                hmap
            }));
        }

        for handle in handles {
            // Wait for the thread to finish. Returns a result.
            let hmap = handle.join();
            match hmap {
                Ok(mut hm) => {
                    for (key, val) in hm.iter_mut() {
                        centroids[*key].indexes.append(val);
                    }
                }
                Err(e) => {panic!("kmeans threading failed {:?}", e)}
            }
        }
        t.stop();
        t.print_as_millis();
        
        // Update
        let mut t = DebugTimer::start("kmeans update");
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
        t.stop();
        t.print_as_millis();
        println!("");
        bar_max_iterations.inc(1);
    }
    bar_max_iterations.finish();
    centroids
}


#[cfg(test)]
mod pq_kmeans_tests {
    use ndarray::{Array2, arr2};
    use crate::algs::kmeans::{kmeans};

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
        let centroids = kmeans(rng, 7, 200, &dataset1().view(), false);
        println!("{:?}", centroids);
        assert!(centroids.len() == 7);
    }

    #[test]
    fn kmeans_with_k_3_seed_111_centroid0_is() {
        use rand::{SeedableRng, rngs::StdRng};
        let rng = StdRng::seed_from_u64(111);
        let centroids = kmeans(rng, 4, 200, &dataset1().view(), false);
        assert!(centroids[0].indexes == vec![3, 4, 5]);
    }
}