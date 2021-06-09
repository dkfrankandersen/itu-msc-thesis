use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use crate::algs::{AlgorithmImpl, distance::cosine_similarity};
use crate::algs::{kmeans::{kmeans}, common::{PQCentroid, Centroid}};
use crate::util::*;

#[derive(Debug, Clone)]
pub struct FAScann {
    name: String,
    metric: String,
    algo_parameters: AlgoParameters,
    dataset: Option<Array2::<f64>>,
    clusters: i32,
    max_iterations: i32,
    clusters_to_search: i32,
    verbose_print: bool
}


impl FAScann {
    pub fn new(verbose_print: bool, algo_parameters: &AlgoParameters, _dataset: &ArrayView2::<f64>, clusters: i32, max_iterations: i32, clusters_to_search: i32) -> Result<Self, String> {
        return Ok(FAScann {
            name: "fa_scann".to_string(),
            metric: "angular".to_string(),
            algo_parameters: algo_parameters.clone(),
            dataset: None,
            clusters: clusters,
            max_iterations: max_iterations,
            clusters_to_search: clusters_to_search,
            verbose_print: verbose_print
        });
    }
}

impl AlgorithmImpl for FAScann {

    fn name(&self) -> String {
        self.name.to_string()
    }

    fn fit(&mut self, dataset: &ArrayView2::<f64>) {
        self.dataset = Some(dataset.to_owned());
    }

    fn query(&self, _dataset: &ArrayView2::<f64>, _p: &ArrayView1::<f64>, results_per_query: usize, arguments: &Vec::<usize>) -> Vec<usize> {
        let _balh = results_per_query;
        let _blah2 = arguments;
        let mut best_n_candidates: Vec<usize> = Vec::new();
        best_n_candidates.reverse();
        if self.verbose_print {
            println!("best_n_candidates \n{:?}", best_n_candidates);
        }
        best_n_candidates
    }

}