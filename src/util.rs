use std::time::{Instant, Duration};
extern crate hdf5;
use hdf5::types::{VarLenUnicode};
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
pub struct Attrs {
    pub algo: VarLenUnicode,
    pub batch_mode: bool,
    pub best_search_time: f64,
    pub build_time: f64,
    pub candidates: f64,
    pub count: u64,
    pub dataset: VarLenUnicode,
    pub distance: VarLenUnicode,
    pub expect_extra: bool,
    pub index_size: f64,
    pub name: VarLenUnicode,
    pub run_count: u64
}

pub fn store_results(results: (Duration, Vec<usize>), attrs: Attrs) -> hdf5::Result<()> {
    let path: &str = "results/";
    let filename: &str = "test4.hdf5";

    let full_path = format!("{}{}", path, filename);
    println!("{}", full_path);
    let _e = hdf5::silence_errors();
    {
        let file = hdf5::File::create(full_path);
        match file {
            Ok(f) => {
                        let attributes = f.new_dataset::<Attrs>().create("attributes", 1)?;
                        attributes.write(&[attrs]);
                        // match attributes {
                        //     Ok(a) => a
                        // }
                        let times = f.new_dataset::<f64>().create("times", 2);
                        let neighbors = f.new_dataset::<i32>().create("neighbors", 2);
                        let distances = f.new_dataset::<f64>().create("distances", 2);
            },
            Err(e) => println!("Error {}", e)
        }
        
    }
    Ok(())
}

