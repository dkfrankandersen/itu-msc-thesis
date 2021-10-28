use ndarray::prelude::*;
// use crate::algs::common::Centroid;

// #[allow(dead_code)]
// pub fn recompute_centroids_simple(spherical: bool, dataset: &ArrayView2::<f64>, centroids: &mut Vec::<Centroid>) {
//     // Update new means
//     for centroid in centroids.iter_mut() {
//         // Clear centroid point
//         centroid.point.fill(0.);

//         if centroid.indexes.len() > 0 {
//             // Add dimension value of each
//             for index in centroid.indexes.iter() {
//                 let point = dataset.slice(s![*index,..]);
//                 for (i, x) in point.iter().enumerate() {
//                     centroid.point[i] += x;
//                 }
//             }

//             let divisor = if spherical { centroid.point.dot(&centroid.point).sqrt() }
//                           else { centroid.indexes.len() as f64};
     
//             if divisor == 0. {
//                 println!("recompute_centroids_simple, could not normalize centroid due to zero norm or empty partition.");
//                 continue;
//             }
//             let multiplier = 1.0 / divisor;
//             // Divide by indexes to get mean
//             for i in 0..centroid.point.len() {  
//                 centroid.point[i] *= multiplier;
//             }
//         }
//     }
// }


// perpendicular_norm_delta =
//         residual_norm_delta - parallel_norm_delta;

// cost_delta = parallel_cost_multiplier * parallel_norm_delta +
//                         perpendicular_norm_delta;

// pub fn _anisotropic_loss(p: ArrayView1::<f64>, q: ArrayView1::<f64>) {
//     let w = 1. as f64;
//     let a = h_parallel(p.dot(&p), w, p.len()) * r_parallel_residual_error(&p, &q);
//     let b = h_orthogonal(p, w) * 
//      //*  + h_orthogonal(w: f64, x: f64)
// }

pub fn check_dimension_eq(a: usize, b: usize, msg: &str) {
    if a != b {
        panic!("Dimension check {}=={} failed in {}", a, b, msg);
    }
}

pub fn check_dimension_ge(a: usize, b: usize, msg: &str) {
    if a != b {
        panic!("Dimension check {}>={} failed in {}", a, b, msg);
    }
}

pub fn squared_l2_norm(p: ArrayView1<f64>) -> f64 {
    let mut res: f64 = 0.0;
    for x in p.iter() {
        res += x*x;
    }
    res.sqrt()
}

// Checked 1:1 OK
#[derive(Debug, Clone)]
pub struct CoordinateDescentResult {
    new_center_idx: usize,
    cost_delta: f64,
    new_parallel_residual_component: f64
}

// Checked 1:1 OK
#[derive(Debug, Clone)]
pub struct SubspaceResidualStats {
    residual_norm: f64,
    parallel_residual_component: f64
}

// Checked 1:1 OK
pub fn compute_residual_stats_for_cluster(
    maybe_residual_dptr: &Vec<f64>, original_dptr: &Vec<f64>,
    inv_norm: f64, quantized: &Vec<f64>) -> SubspaceResidualStats {

    check_dimension_eq(maybe_residual_dptr.len(), original_dptr.len(), "compute_residual_stats_for_cluster");
    let mut result = SubspaceResidualStats{residual_norm: 0.0, parallel_residual_component: 0.0};
    
    for i in 0..maybe_residual_dptr.len() {
        let residual_coordinate: f64 = maybe_residual_dptr[i] - quantized[i];
        result.residual_norm += residual_coordinate.sqrt();
        result.parallel_residual_component +=
            residual_coordinate * original_dptr[i] * inv_norm;
  }
  result
}

pub fn get_chunked_datapoint(point: ArrayView1<f64>, num_subspaces: usize) -> Vec<Vec<f64>>{
    let chunk_size = point.len() / num_subspaces;
    let mut chunked = vec![vec![0.; chunk_size]; num_subspaces];
    let mut index = 0;
    for m in 0..num_subspaces {
        for c in 0..chunk_size {
            chunked[m][c] = point[index];
            index += 1;
        }
    }
    chunked
}

pub fn compute_residual_stats(maybe_residual_dptr: ArrayView1<f64>,  original_dptr: ArrayView1<f64>, centers: &Vec<Vec<Vec<f64>>>) -> Vec<Vec<SubspaceResidualStats>> {
    // let result = Vec<Vec<SubspaceResidualStats>>
    let num_subspaces: usize = centers.len(); // expect m=50
    check_dimension_ge(num_subspaces, 1, "compute_residual_stats");
    
    let num_clusters_per_block = centers[0].len(); // expect k=16
    let mut residual_stats = vec![vec![SubspaceResidualStats {residual_norm: 0.0, parallel_residual_component: 0.0}; num_clusters_per_block]; num_subspaces];
    let maybe_residual_dptr_chunked = get_chunked_datapoint(maybe_residual_dptr, num_subspaces);
    let original_dptr_chunked = get_chunked_datapoint(original_dptr, num_subspaces);
    check_dimension_eq(maybe_residual_dptr_chunked.len(), num_subspaces, "compute_residual_stats");
    check_dimension_eq(original_dptr_chunked.len(), num_subspaces, "compute_residual_stats");

    let  mut chunked_norm: f64 = 0.0;
    for m in 0..num_subspaces {
        for x in original_dptr_chunked[m].iter() {
            chunked_norm += x*x;
        }
    }
    let chunked_norm: f64 = chunked_norm.sqrt();
    let inverse_chunked_norm: f64 = 1.0 / chunked_norm;

    for subspace_idx in 0..num_subspaces { // m
        // let mut cur_subspace_residual_stats = &residual_stats[subspace_idx];
        // let cur_subspace_centers = &centers[subspace_idx];
        
        for cluster_idx in 0..num_clusters_per_block { //  k
            // let center = &cur_subspace_centers[cluster_idx];
            let maybe_residual_dptr_span = &maybe_residual_dptr_chunked[subspace_idx];
            let original_dptr_span = &original_dptr_chunked[subspace_idx];
            residual_stats[subspace_idx][cluster_idx] = compute_residual_stats_for_cluster(
                    &maybe_residual_dptr_span, &original_dptr_span, inverse_chunked_norm,
                    &centers[subspace_idx][cluster_idx]);
        }
    }
    residual_stats
}
 
pub fn optimize_single_subspace(
    cur_subspace_residual_stats: Vec<SubspaceResidualStats>,
    cur_center_idx: usize, parallel_residual_component: f64,
    parallel_cost_multiplier: f64) -> CoordinateDescentResult {

    let mut result = CoordinateDescentResult {
        new_center_idx: cur_center_idx,
        cost_delta: 0.0,
        new_parallel_residual_component: parallel_residual_component};

    let old_subspace_residual_norm: f64 =
        cur_subspace_residual_stats[cur_center_idx].residual_norm;
    let old_subspace_parallel_component: f64 =
        cur_subspace_residual_stats[cur_center_idx].parallel_residual_component;
    for new_center_idx in 0..cur_subspace_residual_stats.len() {
        if new_center_idx == cur_center_idx { continue; }

        let rs: SubspaceResidualStats = cur_subspace_residual_stats[new_center_idx].clone();
        let new_parallel_residual_component: f64 =
            parallel_residual_component - old_subspace_parallel_component +
            rs.parallel_residual_component;
        let parallel_norm_delta: f64 = new_parallel_residual_component.sqrt() -
                                            parallel_residual_component.sqrt();
        if parallel_norm_delta > 0.0 { continue; }
        
        let residual_norm_delta: f64 =
            rs.residual_norm - old_subspace_residual_norm;
        let perpendicular_norm_delta: f64 =
            residual_norm_delta - parallel_norm_delta;
        let cost_delta: f64 = parallel_cost_multiplier * parallel_norm_delta +
                                perpendicular_norm_delta;
        if cost_delta < result.cost_delta {
            result.new_center_idx = new_center_idx;
            result.cost_delta = cost_delta;
            result.new_parallel_residual_component = new_parallel_residual_component;
        }
    }
    return result;
}


pub fn compute_parallel_residual_component(quantized: Vec<usize>, residual_stats: Vec::<Vec::<SubspaceResidualStats>>) -> f64 {
    let mut result: f64 = 0.0;
    for subspace_idx in 0..quantized.len() {
      let cluster_idx: usize = quantized[subspace_idx];
      result +=
          residual_stats[subspace_idx][cluster_idx].parallel_residual_component;
    }
    return result;
}

// pub fn initialize_to_min_residual_norm(residual_stats: Vec<Vec<SubspaceResidualStats>>, result: &mut Vec<usize>) {
//   for subspace_idx in 0..residual_stats.len() {
    
//     let mut min_norm = f64::INFINITY;
//     for x in subspace_idx.iter() {
//         if x.residual_norm < min {
//             min = x.residual_norm;
//         }
//     }
    
//     result[subspace_idx] = (min_norm - residual_stats[subspace_idx][0]) as usize;
//   }
// }

// pub fn coordinate_descent_ah_quantize(
//     maybe_residual_dptr: ArrayView1<f64>,  original_dptr: ArrayView1<f64>,
//     centers: Vec::<Centroid>, const ChunkingProjection<T>& projection, threshold: f64, result: Vec<usize>) {

    // let residual_stats = compute_residual_stats(maybe_residual_dptr, original_dptr, centers, projection);

    // let parallel_cost_multiplier: f64 = compute_parallel_cost_multiplier(
    //                                                     threshold, squared_l2_norm(original_dptr), original_dptr.len());
    //                                                     initialize_to_min_residual_norm(residual_stats, result);

    // let parallel_residual_component: f64 = compute_parallel_residual_component(result, residual_stats);

    // let subspace_idxs = Vec<usize>::new();
    // let subspace_residual_norms = Vec<f64>::new();
    // for subspace_idx in 0..result.len() {
    //     let cluster_idx: usize = result[subspace_idx];
    //     subspace_residual_norms[subspace_idx] =
    //         residual_stats[subspace_idx][cluster_idx].residual_norm;
    // }
    // let result_sorted =  Vec<usize>::new();

    // let k_max_rounds: usize = 10;
    // let cur_round_changes = true;
    // for _ in 0..k_max_rounds {
    //     cur_round_changes = false;
    //     for i in 0..subspace_idxs.len() {
    //         let subspace_idx: usize = subspace_idxs[i];
    //         let cur_subspace_residual_stats: Vec<SubspaceResidualStats> = residual_stats[subspace_idx];
    //         let cur_center_idx: usize = result_sorted[i];
    //         let subspace_result = optimize_single_subspace(
    //             cur_subspace_residual_stats, cur_center_idx,
    //             parallel_residual_component, parallel_cost_multiplier);
    //         if (subspace_result.new_center_idx != cur_center_idx) {
    //             parallel_residual_component =
    //                 subspace_result.new_parallel_residual_component;
    //             result_sorted[i] = subspace_result.new_center_idx;
    //             cur_round_changes = true;
    //         }
    //     }
    //     if cur_round_changes != true {
    //         break
    //     }
    // }

    // let mut final_residual_norm: f64 = 0.0;
    // for i in 0..result_sorted.len() {
    //     let subspace_idx: usize = subspace_idxs[i];
    //     let center_idx: usize = result_sorted[i];
    //     result[subspace_idx] = center_idx;
    //     final_residual_norm +=
    //         residual_stats[subspace_idx][center_idx].residual_norm;
    // }
// }

// pub fn compute_parallel_cost_multiplier(threshold_t: &f64, squared_l2_norm: f64, dim: usize) -> f64 {
//     // ScaNN Paper Theorem 3.4
//     let threshold_t_squared = threshold_t*threshold_t;
//     let parallel_cost: f64 = threshold_t_squared / squared_l2_norm;
//     let perpendicular_cost: f64 = (1.0 - threshold_t_squared / squared_l2_norm) / (dim - 1) as f64;

//     let result = parallel_cost / perpendicular_cost;
//     result
// }

pub fn coordinate_descent_ah_quantize(residual: ArrayView1::<f64>,  datapoint: ArrayView1<f64>,
                                                 centers: &Vec::<Array1::<f64>>, threshold: &f64) {
    // println!("Dimension residual.len() = {} expect D=100", residual.len());
    // println!("Dimension datapoint.len() = {} expect D=100", datapoint.len());
    // println!("Dimension centers.len() = {} expect M=50", centers.len());
    // println!("Dimension centers[0].len() = {} expect K=16", centers[0].len());
    // panic!("ARHHHHH");
    // let residual_stats: Vec<Vec<SubspaceResidualStats>> = compute_residual_stats(residual, datapoint, centers);
    // let parallel_cost_multiplier: f64 = compute_parallel_cost_multiplier(&threshold, squared_l2_norm(datapoint), datapoint.dim());
    // // let result:Vec<usize>  = InitializeToMinResidualNorm(residual_stats, result);

    
}