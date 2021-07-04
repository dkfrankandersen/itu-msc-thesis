use ndarray::{Array, Array1, Array2, s, ArrayView1};
use std::collections::{HashMap};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct Centroid {
    pub id: usize,
    pub point: Array1<f64>,
    pub indexes: Vec::<usize>
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct PQCentroid {
    pub id: usize,
    pub point: Array1<f64>,
    pub children: HashMap::<usize, Vec::<usize>>
}

impl PQCentroid {
    pub fn compute_residual(&self, query: &ArrayView1<f64>) -> Array1<f64> {
        // Compute residuals between query and coarse_quantizer
        query-&self.point
    }

    pub fn compute_distance_table(&self, residual_point: &Array1::<f64>, residuals_codebook: &Array2::<Array1::<f64>>) -> Array2::<f64> {
        // Create a distance table, for each of the M blocks to all of the K codewords -> table of size M times K.
        let m_dim = residuals_codebook.nrows();
        let k_dim = residuals_codebook.ncols();
        let mut distance_table = Array::from_elem((m_dim, k_dim), 0.);
        let dim = residual_point.len()/m_dim;
        for m in 0..m_dim {

            let begin = dim * m;
            let end = begin + dim;
            
            let partial_query = residual_point.slice(s![begin..end]);
            for k in 0..k_dim {
                let partial_residual_codeword = residuals_codebook[[m, k]].view();
                distance_table[[m,k]] = partial_residual_codeword.dot(&partial_query);
            }
        }
        distance_table
    }

    pub fn distance_from_indexes(&self, distance_table: &Array2::<f64>, child_values: &Vec::<usize>) -> f64 {
        let mut distance: f64 = 0.;
        for (m, k) in child_values.iter().enumerate() {
            distance += &distance_table[[m, *k]];
        }
        distance
    }
}