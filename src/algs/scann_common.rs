use ndarray::prelude::*;

pub fn debug_track_query_top_results(index: &usize, msg: String) {
    // let neighbors = vec![
    //                         // glove-100-angular.hdf5
    //                         // 97478,  262700,  846101,  671078,  232287,  727732,  544474,
    //                         // 1133489,  723915,  660281
    //                         // ,  566421, 1093917,  908319,  656605,
    //                         // 93438,  326455,  584887, 1096614,  100206,  547334,  674655,
    //                         // 834699,  445577,  979282,  776528,   51821,  994865,  186281,
    //                         // 533888,  331310, 1037752,  193057,  859959,  368655,  690267,
    //                         // 82685,  484525, 1168162, 1069248, 1126839,  256447,  451625,
    //                         // 914908,  873104,  956338,  678395,  939324,  748511,  207076,
    //                         // 751282,  817757,  402216,  932395,  290452,  265744,  696453,
    //                         // 82910,  436049,  712479,  494528,  989330,  655775,  995275,
    //                         // 647843,  375237,  242797, 1116578,  793170,  325682,  265226,
    //                         // 888453,  599119,  631740,  212807, 1142011,  530481,  656064,
    //                         // 944910,  459704,  490937,  239304,  264602,  495380,  843410,
    //                         // 724903,  876802,  636623,  172030,  162588,  761652,   74880,
    //                         // 418892,  687317, 1008844, 1011545,  983601,  340497,  598329,
    //                         // 944409,  725625

    //                         // random-xs-20-angular.hdf5
    //                         3618, 8213, 4462, 6709, 3975, 3129, 5120, 2979, 6319, 3244,
    //                         //   381,
    //                         // 5332, 5846,  319, 3325, 1882, 4401, 2044, 7224, 1294, 7539, 5321,
    //                         // 3247, 5398,   33, 8582, 7254, 1397, 5700, 4536, 2615, 7802, 3220,
    //                         // 4717, 5082, 6604, 2583, 8871, 2275, 4235,  655, 8254, 7007,  511,
    //                         // 3502, 4826, 5959,  533, 8705, 8201, 8054, 5335, 7155, 3313, 2820,
    //                         // 3974,  185, 5523,  839, 6242, 3192, 2180, 2740, 1477,  992, 3602,
    //                         // 3113, 2747, 6137, 5837, 1630,  345, 5159, 8732, 6615, 4195,  325,
    //                         // 2969, 8426,  197, 1064, 5957,  647, 1281, 7618, 5121, 6835, 7551,
    //                         // 7102, 4981, 6960, 1153, 3357, 1479,  564, 6526, 4545, 6335, 1001,
    //                         // 1113
    //                         ];
    // if neighbors.contains(index) {
    //     println!("Found {}:  {}", index, msg);
    // }
}

pub fn check_dimension_eq(a: usize, b: usize, msg: &str) {
    if a != b {
        panic!("Dimension check {}=={} failed in {}", a, b, msg);
    }
}

pub fn check_dimension_ge(a: usize, b: usize, msg: &str) {
    if a < b {
        panic!("Dimension check {}>={} failed in {}", a, b, msg);
    }
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
    maybe_residual_dptr: &[f64], original_dptr: &[f64],
    inv_norm: f64, quantized: &[f64]) -> SubspaceResidualStats {
    // println!("maybe_residual_dptr {:?}", maybe_residual_dptr);
    // println!("original_dptr {:?}", original_dptr);
    // println!("inv_norm {:?}", inv_norm);
    // println!("quantized {:?}", quantized);


    check_dimension_eq(maybe_residual_dptr.len(), original_dptr.len(), "compute_residual_stats_for_cluster");
    let mut result = SubspaceResidualStats{residual_norm: 0.0, parallel_residual_component: 0.0};
    
    for i in 0..maybe_residual_dptr.len() {
        let residual_coordinate: f64 = maybe_residual_dptr[i] - quantized[i];
        result.residual_norm += square(residual_coordinate);
        result.parallel_residual_component +=
            residual_coordinate * original_dptr[i] * inv_norm;
    }
    // println!("result.residual_norm {:?}", result.residual_norm);
  result
}

// My understanding of function
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

// My understanding of function
pub fn squared_l2_norm(p: ArrayView1<f64>) -> f64 {
    let mut res: f64 = 0.0;
    for x in p.iter() {
        res += x*x;
    }
    res.sqrt()
}

pub fn square(a: f64) -> f64 {
    a*a
}

// Checked 1:1 OK
pub fn compute_residual_stats(maybe_residual_dptr: ArrayView1<f64>,  original_dptr: ArrayView1<f64>, centers: &[Vec<Vec<f64>>]) -> Vec<Vec<SubspaceResidualStats>> {
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
            chunked_norm += square(*x);
        }
    }
    let chunked_norm: f64 = chunked_norm.sqrt();
    let inverse_chunked_norm: f64 = 1.0 / chunked_norm;

    for subspace_idx in 0..num_subspaces { // m        
        for cluster_idx in 0..num_clusters_per_block { //  k
            let center = &centers[subspace_idx][cluster_idx];
            let maybe_residual_dptr_span = &maybe_residual_dptr_chunked[subspace_idx];
            let original_dptr_span = &original_dptr_chunked[subspace_idx];
            residual_stats[subspace_idx][cluster_idx] = compute_residual_stats_for_cluster(
                    maybe_residual_dptr_span, original_dptr_span, inverse_chunked_norm,
                    center);
        }
    }
    residual_stats
}

// Checked 1:1 OK
pub fn compute_parallel_cost_multiplier(threshold_t: &f64, squared_l2_norm: f64, dimension: usize) -> f64 {
    // ScaNN Paper Theorem 3.4
    let threshold_t_squared = threshold_t*threshold_t;
    let parallel_cost: f64 = threshold_t_squared / squared_l2_norm;
    let perpendicular_cost: f64 = ((1.0 - threshold_t_squared) / squared_l2_norm) / (dimension - 1) as f64;
    parallel_cost / perpendicular_cost
}

// Checked 1:1 OK
pub fn initialize_to_min_residual_norm(residual_stats: &[Vec::<SubspaceResidualStats>], result: &mut Vec<usize>) {
    check_dimension_eq(result.len(), residual_stats.len(), "initialize_to_min_residual_norm");

    let num_subspaces = residual_stats.len();
    let num_clusters_per_block = residual_stats[0].len();
    
    for subspace_idx in 0..num_subspaces { // m        
        let mut min_norm = (f64::INFINITY, 0_usize);
        for cluster_idx in 0..num_clusters_per_block { //  k
            let residual_norm = residual_stats[subspace_idx][cluster_idx].residual_norm;
            if residual_norm < min_norm.0 {
                min_norm = (residual_norm, cluster_idx)
            }
        }
        result[subspace_idx] = min_norm.1;
  }
}

// Checked 1:1 OK
pub fn compute_parallel_residual_component(quantized: &[usize], residual_stats: &[Vec::<SubspaceResidualStats>]) -> f64 {
    let mut result: f64 = 0.0;
    for subspace_idx in 0..quantized.len() {
      let cluster_idx: usize = quantized[subspace_idx];
      result += residual_stats[subspace_idx][cluster_idx].parallel_residual_component;
    }
    result
}

// Checked 1:1 OK
pub fn optimize_single_subspace(
            cur_subspace_residual_stats: &[SubspaceResidualStats],
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
        let parallel_norm_delta: f64 = square(new_parallel_residual_component) -
                                            square(parallel_residual_component);
        if parallel_norm_delta > 0.0 { continue; }
    
        let residual_norm_delta: f64 = rs.residual_norm - old_subspace_residual_norm;
        let perpendicular_norm_delta: f64 = residual_norm_delta - parallel_norm_delta;
        let cost_delta: f64 = parallel_cost_multiplier * parallel_norm_delta +
                                perpendicular_norm_delta;
        if cost_delta < result.cost_delta {
            result.new_center_idx = new_center_idx;
            result.cost_delta = cost_delta;
            result.new_parallel_residual_component = new_parallel_residual_component;
        }
    }
    result
}

// Sorting in the same way as scann do, might be a faster solution
pub fn sorted_by_max_residual_norms(subspace_residual_norms: &mut Vec<f64>, result_sorted: &mut Vec<usize>, subspace_idxs: &mut Vec<usize>) {
    let mut sorted_tuple: Vec<(f64, usize, usize)> = Vec::new();
    for i in 0..result_sorted.len() {
        sorted_tuple.push((subspace_residual_norms[i], result_sorted[i], i));
    }
    sorted_tuple.sort_by(|a, b| b.partial_cmp(a).unwrap());
    for i in 0..result_sorted.len() {
        subspace_residual_norms[i] = sorted_tuple[i].0;
        result_sorted[i] = sorted_tuple[i].1;
        subspace_idxs[i] = sorted_tuple[i].2;
    }
}

pub fn coordinate_descent_ah_quantize(index: &usize, maybe_residual_dptr: ArrayView1<f64>, original_dptr: ArrayView1<f64>,
                                                 centers: &[Vec<Vec<f64>>], threshold: &f64) -> Vec<usize> {
    
    let mut result = vec![0; centers.len()];
    check_dimension_eq(result.len(), centers.len(), "coordinate_descent_ah_quantize");
    check_dimension_eq(maybe_residual_dptr.len(), original_dptr.len(), "coordinate_descent_ah_quantize");

    let residual_stats: Vec<Vec<SubspaceResidualStats>> =
                compute_residual_stats(maybe_residual_dptr, original_dptr, centers);
    
    let parallel_cost_multiplier: f64 =
                compute_parallel_cost_multiplier(threshold, squared_l2_norm(original_dptr), original_dptr.len());
                
    initialize_to_min_residual_norm(&residual_stats, &mut result); // update result with pq codes
       
    let mut parallel_residual_component: f64 = 
                compute_parallel_residual_component(&result, &residual_stats);
    
    let mut subspace_residual_norms = vec![0.0_f64; result.len()];
    let mut result_sorted = result.clone();
    let mut subspace_idxs: Vec<usize> = (0..result.len()).collect();
    
    for subspace_idx in 0..result.len() {
        let cluster_idx = result[subspace_idx];
        subspace_residual_norms[subspace_idx] = residual_stats[subspace_idx][cluster_idx].residual_norm;
    }
    
    sorted_by_max_residual_norms(&mut subspace_residual_norms, &mut result_sorted, &mut subspace_idxs);

    let num_subspaces = result.len();
    let k_max_rounds = 10;
    let mut cur_round_changes = true;
    for _ in 0..k_max_rounds {
        if !cur_round_changes {
            break;
        }
        cur_round_changes = false;
        for i in 0..num_subspaces {
            let subspace_idx = subspace_idxs[i];                
            let cur_subspace_residual_stats = &residual_stats[subspace_idx];
            
            let cur_center_idx: usize = result_sorted[subspace_idx];
            let subspace_result: CoordinateDescentResult =  
                                    optimize_single_subspace(
                                        cur_subspace_residual_stats, cur_center_idx,
                                        parallel_residual_component, parallel_cost_multiplier);
            
            if subspace_result.new_center_idx != cur_center_idx {
                parallel_residual_component = subspace_result.new_parallel_residual_component;
                result_sorted[subspace_idx] = subspace_result.new_center_idx;
                cur_round_changes = true;
            }
        }
    }

    for i in 0..result_sorted.len() {
        let subspace_idx: usize = subspace_idxs[i];
        let center_idx: usize = result_sorted[i];
        result[subspace_idx] = center_idx;
    }
    result
}