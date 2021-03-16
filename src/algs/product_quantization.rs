use crate::algs::*;

#[derive(Debug, Clone)]
struct ProductQuantization {
    name: String,
    metric: String,
    dataset: Option<Array2::<f64>>,
}

impl ProductQuantization {
    pub fn new(codebooks: i32, codewords: i32) -> Self {
        ProductQuantization {

        }
    }

    fn init(&mut self, dataset: &ArrayView2::<f64>) { 

    }

    fn assign(&mut self) { 

    }

    fn update(&mut self) { 

    }

    fn run_pq(&mut self, max_iterations: i32, dataset: &ArrayView2::<f64>) {
        loop {
            if iterations == 1 || iterations % 10 == 0 {
                println!("Iteration {}", iterations);
            }
            if iterations > max_iterations {
                println!("Max iterations reached, iterations: {}", iterations-1);
                break;
            } else if self.codebook == last_codebook {
                println!("Computation has converged, iterations: {}", iterations-1);
                break;
            }

            self.assign();
            self.update();
            iterations += 1;
        }
    }
}

impl AlgorithmImpl for ProductQuantization {

    fn __str__(&self) {
        self.name.to_string();
    }

    fn done(&self) {}

    fn get_memory_usage(&self) {}

    fn fit(&mut self, dataset: ArrayView2::<f64>) {
        self.dataset = Some(dataset.to_owned());
        self.run_pq(self.max_iterations, &dataset);
        
    }

    fn batch_query(&self) {}

    fn get_batch_results(&self) {}
    
    fn get_additional(&self) {
        
    }

    fn query(&self, p: &ArrayView1::<f64>, result_count: u32) -> Vec<usize> {
        
        let mut best_n_candidates: Vec<usize> = Vec::new();
        best_n_candidates.reverse();
        println!("best_n_candidates \n{:?}", best_n_candidates);
        best_n_candidates
    }

}