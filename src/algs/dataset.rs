use ndarray::{Array2, ArrayView1, Array1};

#[derive(Debug)]
pub struct Dataset {
    pub filename: String,
    hdf5_file: hdf5::File
}

impl Dataset {

    pub fn new(filename: &str) -> Dataset {
        Dataset {
            filename: filename.to_string(),
            hdf5_file: hdf5::File::open(filename.to_string()).unwrap()
        }
    }

    pub fn get(&self, dataset: &str) -> Array2::<f64> {
        return (self.hdf5_file).dataset(dataset).unwrap().read_2d::<f64>().unwrap();
    }

    pub fn get_as_usize(&self, dataset: &str) -> Array2::<usize> {
        return (self.hdf5_file).dataset(dataset).unwrap().read_2d::<usize>().unwrap();
    }

    pub fn get_as_normalize(&self, dataset: &str) -> Array2::<f64> {
        let dataset = (self.hdf5_file).dataset(dataset).unwrap().read_2d::<f64>().unwrap();
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
}