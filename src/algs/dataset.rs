use ndarray::{Array2, ArrayView1, Array1};

#[derive(Debug)]
pub struct Dataset {
    pub filename: String,
    hdf5_file: hdf5::File
}


#[allow(dead_code)]
impl Dataset {

    pub fn new(filename: &str) -> Self {
        Dataset {
            filename: filename.to_string(),
            hdf5_file: hdf5::File::open(filename.to_string()).unwrap()
        }
    }

    pub fn get_f64(&self, dataset: &str) -> Array2::<f64> {
        return (self.hdf5_file).dataset(dataset).unwrap().read_2d::<f64>().unwrap();
    }

    pub fn get_usize(&self, dataset: &str) -> Array2::<usize> {
        return (self.hdf5_file).dataset(dataset).unwrap().read_2d::<usize>().unwrap();
    }

    pub fn neighbors(&self) -> Array2::<usize> {
        return self.get_usize("neighbors");
    }

    pub fn train_normalize(&self) -> Array2::<f64> {
        return self.get_as_normalize("train");
    }

    pub fn test_normalize(&self) -> Array2::<f64> {
        return self.get_as_normalize("test");
    }

    pub fn distances_normalize(&self) -> Array2::<f64> {
        return self.get_as_normalize("distances");
    }

    pub fn get_as_normalize(&self, dataset: &str) -> Array2::<f64> {
        let dataset = self.get_f64(dataset);
        let mut ds_new: Array2<f64> = dataset.clone();
        for (idx_row, p) in dataset.outer_iter().enumerate() {
            let magnitude = p.dot(&p).sqrt();        
            for (idx_col, val) in p.iter().enumerate() {
                ds_new[[idx_row, idx_col]] = val/magnitude;
            }       
        }
        return ds_new;
    }

    pub fn normalize_all(&self, dataset: Array2<f64>) -> Array2::<f64> {
        let mut ds_new: Array2<f64> = dataset.clone();
        for (idx_row, p) in dataset.outer_iter().enumerate() {
            let magnitude = p.dot(&p).sqrt();        
            for (idx_col, val) in p.iter().enumerate() {
                ds_new[[idx_row, idx_col]] = val/magnitude;
            }            
        }
        return ds_new;
    }
    
    pub fn normalize_vector(p: &ArrayView1::<f64>)-> Array1::<f64>  {
        let magnitude = p.dot(p).sqrt();
        return p.map(|e| e/magnitude);
    }

    pub fn print_true_neighbors(&self, from : usize, to: usize, m: usize) {
        let dataset = self.neighbors();
        println!("| Distance for {} closests neighbors from {} to {}:", m, from, to);
        for i in from..to {
            let mut neighbors = Vec::new();
            for j in 0..m {
                neighbors.push(dataset[[i,j]]);
            }
            println!("|  idx: {} neighbors {:?}", i, neighbors);
        }
        println!("");
    }
}