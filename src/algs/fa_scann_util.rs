use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, s};
use std::collections::{HashMap};
use serde::{Serialize, Deserialize};

fn r_parallel_residual_error(x: &ArrayView1::<f64>, q: &ArrayView1::<f64>) -> Array1::<f64> {
    // Takes dot product of the residuals (x-q) and x, then multiplie onto x and divides with the norm of x to the power of 2 (so just dot product).
    ((x-q).dot(x)*x) / x.dot(x) //.sqrt().powi(2)
}

fn r_orthogonal_residual_error(x: &ArrayView1::<f64>, q: &ArrayView1::<f64>) -> Array1::<f64> {
    (x-q) - r_parallel_residual_error(x, q)
}

fn h_parallel(x:f64, w: f64, d: usize) -> f64 {
    panic!("todo");
}

fn h_orthogonal(x: f64, w: f64) -> f64 {
    panic!("todo");
}

fn eta_value(threshold_t: f64, dimension: usize) -> f64 {
    threshold_t.powi(2) / (1.-threshold_t.powi(2)) * ((dimension-1) as f64)
}

#[cfg(test)]
mod fa_scann_util_tests {
    use crate::algs::fa_scann_util::*;
    use assert_float_eq::*;

    #[test]
    fn eta_value_with_t_0_2_and_d_100_return_4_125() {
        let eta = eta_value(0.2, 100);
        println!("{:?}", eta);
        assert!(expect_f64_near!(eta, 4.125 as f64).is_ok());
    }

    #[test]
    fn eta_value_with_t_0_and_d_100_return_0() {
        let eta = eta_value(0.0, 100);
        println!("{:?}", eta);
        assert!(expect_f64_near!(eta, 0. as f64).is_ok());
    }

    #[test]
    fn eta_value_with_t_1_and_d_100_return_4_125() {
        let eta = eta_value(1.0, 100);
        println!("{:?}", eta);
        assert!(expect_f64_near!(eta, f64::INFINITY).is_ok());
    }
}

fn score_aware_quantization_loss_fixed_eta(p: &ArrayView1::<f64>, q: &ArrayView1::<f64>, w: f64) {

}



// double ComputeParallelCostMultiplier(double t, double squared_l2_norm, DimensionIndex dims) {
//     const double parallel_cost = Square(t) / squared_l2_norm;
//     const double perpendicular_cost =
//     (1.0 - Square(t) / squared_l2_norm) / (dims - 1.0);
//     return parallel_cost / perpendicular_cost;
// }

pub fn _compute_parallel_cost_multiplier(t: f64, squared_l2_norm: f64, dim: usize) -> f64 {
    // ScaNN Paper Theorem 3.4
    let parallel_cost: f64 = t.sqrt() / squared_l2_norm;
    let perpendicular_cost: f64 = (1.0 - t.sqrt()) / squared_l2_norm / (dim - 1) as f64;

    let result = parallel_cost / perpendicular_cost;
    result
}

// pub fn _anisotropic_loss(p: ArrayView1::<f64>, q: ArrayView1::<f64>) {
//     let w = 1. as f64;
//     let a = h_parallel(p.dot(&p), w, p.len()) * r_parallel_residual_error(&p, &q);
//     let b = h_orthogonal(p, w) * 
//      //*  + h_orthogonal(w: f64, x: f64)
// }

// pub fn blah(child_vectors: Vec<Array2::<f64>>, eta: f64) -> Array2::<f64> {
//     let m = child_vectors.len();

//     let qunatizer_point = Array::from_elem()

// }

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct Centroid {
    pub id: usize,
    pub point: Array1<f64>,
    pub indexes: Vec::<usize>
}

pub fn recompute_centroids_simple(spherical: bool, dataset: ArrayView2::<f64>, datapoint_dimension: usize, parallel_cost_multiplier: f64, centroids: &mut Vec::<Centroid>) {
    // Update new means
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

            let divisor = if spherical { centroid.point.dot(&centroid.point).sqrt() }
                          else { centroid.indexes.len() as f64};
     
            if divisor == 0. {
                continue;
            }
            let multiplier = 1.0 / divisor;
            // Divide by indexes to get mean
            for i in 0..datapoint_dimension {  
                centroid.point[i] *= multiplier;
            }
        }
    }
}

// pub fn recompute_centroids_with_parallel_cost_multiplier(centroids: &mut Vec::<Centroid>, dataset: ArrayView2::<f64>, datapoint_dim: usize, parallel_cost_multiplier: f64, centroids: Vec::<Centroid>) {
//     extern crate nalgebra as na;
//     use na::{LU, DVector, DMatrix};

//     let parallel_cost_multiplier: f64 = parallel_cost_multiplier;
//     if parallel_cost_multiplier == 1.0 {
//         panic!("parallel_cost_multiplier is 1.0, should be something else");
//     }
//     let dimensionality: usize = datapoint_dim;

//     let mean_vec = Vec::<f64>::with_capacity(dimensionality);

//     let mut means = Vec::<Vec::<f64>>::new();
//     recompute_centroids_simple(false, dataset, dimensionality, parallel_cost_multiplier, &mut centroids);

//     fn add_outer_product(vec: Vec::<f64>) {
//         let outer_prodsums::Eigen
//         let vm = DVector::from_vec(vec);
//         let denom = vm.transpose() * vm;
//         if denom > 0. {
//             outer_prodsums += (vm * vm.transpose()) / denom;
//         }
//     }
    


// }

/***
 * Status GmmUtils::RecomputeCentroidsWithParallelCostMultiplier(
    ConstSpan<pair<uint32_t, double>> top1_results, GmmUtilsImplInterface* impl,
    ConstSpan<uint32_t> partition_sizes, bool spherical,
    DenseDataset<double>* centroids) const {
  const double parallel_cost_multiplier = opts_.parallel_cost_multiplier;
  SCANN_RET_CHECK_NE(parallel_cost_multiplier, 1.0);
  const size_t dimensionality = impl->dimensionality();

  vector<double> mean_vec(centroids->data().size());
  DenseDataset<double> means(std::move(mean_vec), centroids->size());
  SCANN_RETURN_IF_ERROR(RecomputeCentroidsSimple(
      top1_results, impl, partition_sizes, false, &means));

  vector<std::vector<DatapointIndex>> assignments(centroids->size());
  for (DatapointIndex dp_idx : IndicesOf(top1_results)) {
    const size_t partition_idx = top1_results[dp_idx].first;
    assignments[partition_idx].push_back(dp_idx);
  }

  Datapoint<double> storage;
  Eigen::MatrixXd outer_prodsums;
  auto add_outer_product = [&outer_prodsums](ConstSpan<double> vec) {
    DCHECK_EQ(vec.size(), outer_prodsums.cols());
    DCHECK_EQ(vec.size(), outer_prodsums.rows());
    Eigen::Map<const Eigen::VectorXd> vm(vec.data(), vec.size());
    const double denom = vm.transpose() * vm;
    if (denom > 0) {
      outer_prodsums += (vm * vm.transpose()) / denom;
    }
  };
  const double lambda = 1.0 / parallel_cost_multiplier;
  for (size_t partition_idx : IndicesOf(assignments)) {
    MutableSpan<double> mut_centroid = centroids->mutable_data(partition_idx);
    if (assignments[partition_idx].empty()) {
      std::fill(mut_centroid.begin(), mut_centroid.end(), 0.0);
      continue;
    }
    outer_prodsums = Eigen::MatrixXd::Zero(dimensionality, dimensionality);
    for (DatapointIndex dp_idx : assignments[partition_idx]) {
      ConstSpan<double> datapoint =
          impl->GetPoint(dp_idx, &storage).values_slice();
      add_outer_product(datapoint);
    }
    outer_prodsums *= (1.0 - lambda) / assignments[partition_idx].size();

    for (size_t i : Seq(dimensionality)) {
      outer_prodsums(i, i) += lambda;
    }

    Eigen::Map<const Eigen::VectorXd> mean(means[partition_idx].values(),
                                           dimensionality);
    Eigen::VectorXd centroid = outer_prodsums.inverse() * mean;
    for (size_t i : Seq(dimensionality)) {
      mut_centroid[i] = centroid(i);
    }
  }
  return OkStatus();
}
 */