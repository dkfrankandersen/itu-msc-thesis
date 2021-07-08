use crate::algs::scann_impl::common::*;
use ndarray::prelude::*;
use ndarray_linalg::*;

#[allow(dead_code)]
pub fn recompute_centroids_simple(spherical: bool, dataset: &ArrayView2::<f64>, centroids: &mut Vec::<Centroid>) {
    // Update new means
    for centroid in centroids.iter_mut() {
        if centroid.indexes.len() > 0 {
            // Clear centroid point
            centroid.point.fill(0.);

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

#[allow(dead_code)]
pub fn add_outer_product(outer_prodsums: Array2::<f64>, vec: Array1::<f64>) -> Array2::<f64> {
    let mut outer_prodsums = outer_prodsums.clone();
    if vec.len() != outer_prodsums.nrows() {
        panic!("add_outer_product row dimension dont match");
    }
    if vec.len() != outer_prodsums.ncols() {
        panic!("add_outer_product column dimension dont match");
    }
    
    let denom: f64 = vec.dot(&vec);
    if denom > 0. {
        let mut matrix: Array2::<f64> = Array::from_elem((2,vec.len()), 0.);
        matrix.row_mut(0).assign(&vec);
        let nominator = &matrix.t().dot(&matrix);
        let res = nominator / denom;
        outer_prodsums = outer_prodsums + res;
    }
    outer_prodsums
}

#[allow(dead_code)]
pub fn recompute_centroids_with_parallel_cost_multiplier(centroids: &mut Vec::<Centroid>, dataset: &ArrayView2::<f64>, parallel_cost_multiplier: f64) {
    println!("recompute_centroids_with_parallel_cost_multiplier, parallel_cost_multiplier {}", parallel_cost_multiplier);
    if parallel_cost_multiplier == 1.0 {
        panic!("parallel_cost_multiplier is 1.0, should be something else");
    }
    let dimensionality: usize = dataset.ncols();

    recompute_centroids_simple(false, &dataset, centroids);

    let mut outer_prodsums = Array2::from_elem((dimensionality, dimensionality), 0.);

    let lambda = 1.0 / parallel_cost_multiplier;
    for centroid in centroids.iter_mut() {
        let mean = centroid.point.clone();
        if centroid.indexes.len() == 0 {
            centroid.point.fill(0.);
            continue;
        }

        outer_prodsums.fill(0.);
        for index in centroid.indexes.iter() {
            let index_point = &dataset.slice(s![*index,..]).clone();
            outer_prodsums = add_outer_product(outer_prodsums, index_point.to_owned());
        }
        outer_prodsums *= (1.0 - lambda) / centroid.indexes.len() as f64;

        for i in 0..dimensionality {
            outer_prodsums[[i,i]] += lambda;
        }

        match outer_prodsums.inv() {
            Ok(outer_prodsums_inv) => {
                let new_point: Array1::<f64> = outer_prodsums_inv.dot(&mean.t());
                centroid.point = new_point;
            },
            Err(e) => {panic!("Outersum not an invertible matrix\n{}", e)}
        }
        
    }
}