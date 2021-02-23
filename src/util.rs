use std::time::{Instant, Duration};
extern crate hdf5;
use hdf5::types::{VarLenUnicode};
use std::str::FromStr;

// pub struct Attrs {
//     'algo': 'annoy',
//     'batch_mode': False,
//     'best_search_time': 0.00016217617988586425,
//     'build_time': 388.5926456451416,
//     'candidates': 10.0,
//     'count': 10,
//     'dataset': 'glove-100-angular',
//     'distance': 'angular',
//     'expect_extra': False,
//     'index_size': 2705648.0,
//     'name': 'Annoy(n_trees=100,search_k=100)',
//     'run_count': 3
// }

#[derive(hdf5::H5Type, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct AttributesForH5 {
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

#[derive(Debug)]
pub struct Attributes {
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

pub fn store_results(results: Vec<(f64, std::vec::Vec<(usize, f64)>)>, attrs: Attributes) -> hdf5::Result<()> {
    let path: &str = "results/";
    let filename: &str = "test5.hdf5";

    let full_path = format!("{}{}", path, filename);
    println!("{}", full_path);
    let _e = hdf5::silence_errors();
    {
        let file = hdf5::File::create(full_path);
        match file {
            Ok(f) => {
                        let attributes = f.new_dataset::<AttributesForH5>().create("attributes", 1)?;
                        attributes.write(&[attrs.get_as_h5()]);
                        let times = f.new_dataset::<f64>().create("times", 2);
                        let neighbors = f.new_dataset::<i32>().create("neighbors", 2);

                        let distances = f.new_dataset::<f64>().create("distances", 2);
                        
            },
            Err(e) => println!("Error {}", e)
        }
        
    }
    Ok(())
}