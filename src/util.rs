use std::time::{Instant, Duration};
extern crate hdf5;
use hdf5::types::{VarLenUnicode};
use std::str::FromStr;
use ndarray::{s, Array1};

#[derive(hdf5::H5Type, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct AttributesForH5 {
    pub algo: VarLenUnicode,
    pub batch_mode: bool,
    pub best_search_time: f64,
    pub build_time: f64,
    pub candidates: f64,
    pub count: u32,
    pub dataset: VarLenUnicode,
    pub distance: VarLenUnicode,
    pub expect_extra: bool,
    pub index_size: f64,
    pub name: VarLenUnicode,
    pub run_count: u32
}

#[derive(Debug, Clone)]
pub struct Attributes {
    pub algo: String,
    pub batch_mode: bool,
    pub best_search_time: f64,
    pub build_time: f64,
    pub candidates: f64,
    pub count: u32,
    pub dataset: String,
    pub distance: String,
    pub expect_extra: bool,
    pub index_size: f64,
    pub name: String,
    pub run_count: u32
}

impl Attributes {
    pub fn get_as_h5(&self) -> AttributesForH5 {
        AttributesForH5 {
            algo: VarLenUnicode::from_str(&self.algo).unwrap(),
            batch_mode: self.batch_mode,
            best_search_time: self.best_search_time,
            build_time: self.build_time,
            candidates: self.candidates,
            count: self.count,
            dataset: VarLenUnicode::from_str(&self.dataset).unwrap(),
            distance: VarLenUnicode::from_str(&self.distance).unwrap(),
            expect_extra: self.expect_extra,
            index_size: self.index_size,
            name: VarLenUnicode::from_str(&self.name).unwrap(),
            run_count: self.run_count
        }
    }
}

fn get_result_filename(path: &str, attrs: Attributes) -> String {
    format!("{:?}/{:?}_{:?}_{:?}", "", path, attrs.dataset, attrs.count)
}

pub fn store_results(results: Vec<(f64, std::vec::Vec<(usize, f64)>)>, attrs: Attributes) -> hdf5::Result<()> {
    let path: &str = "results/";
    let filename: &str = "test20.hdf5"; 

    let full_path = format!("{}{}", path, filename);
    println!("{}", full_path);
    // let full_path2 = get_result_filename("results", attrs);
    // println!("{}", full_path2);
    let _e = hdf5::silence_errors();
    {
        let file = hdf5::File::create(full_path);
        match file {
            Ok(f) => {
                        let attributes = f.new_dataset::<AttributesForH5>().create("attributes", 1)?;
                        attributes.write(&[attrs.get_as_h5()]).ok();
                        
                        let times = f.new_dataset::<f64>().create("times", results.len())?;
                        // let result_count = attrs.count as usize;
                        let result_count = 10;
                        let neighbors = f.new_dataset::<i32>().create("neighbors", (results.len(), result_count))?;
                        let distances = f.new_dataset::<f64>().create("distances", (results.len(), result_count))?;                    
                         
                        let mut res_times: Vec<f64> = Vec::new();                       
                        for (i, (time, result)) in results.iter().enumerate() { 
                            res_times.push(*time); 
                            let (res_neigh, res_dist): (Vec<usize>, Vec<f64>) = result.iter().cloned().unzip();
                            neighbors.write_slice(&res_neigh, s![i,..]).ok();
                            distances.write_slice(&res_dist, s![i,..]).ok();
                        }
                        times.write(&res_times).ok();
            },
            Err(e) => println!("Error {}", e)
        }
    }
    Ok(())
}

