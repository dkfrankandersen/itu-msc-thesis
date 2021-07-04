use hdf5::types::{VarLenUnicode};
use std::str::FromStr;
use ndarray::{s};
use std::fs;

#[derive(Debug, Clone)]
pub struct Attributes {
    pub batch_mode: bool,
    pub best_search_time: f64,
    pub candidates: f64,
    pub expect_extra: bool,
    pub name: String,
    pub run_count: u32,
    pub distance: String,
    pub count: u32,
    pub build_time: f64,
    pub index_size: f64,
    pub algo: String,
    pub dataset: String,
}

#[derive(hdf5::H5Type, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct AttributesForH5 { 
    pub batch_mode: VarLenUnicode,
    pub best_search_time: VarLenUnicode,
    pub candidates: VarLenUnicode,
    pub expect_extra: VarLenUnicode,
    pub name: VarLenUnicode,
    pub run_count: VarLenUnicode,
    pub distance: VarLenUnicode,
    pub count: VarLenUnicode,
    pub build_time: VarLenUnicode,
    pub index_size: VarLenUnicode,
    pub algo: VarLenUnicode,
    pub dataset: VarLenUnicode,
}

// Quick fix because HDF5-Rust does not work well with attributes yet, so need a Python script to fix it.
pub fn key_type_data(key: String, datatype: String, data: String) -> VarLenUnicode {
    VarLenUnicode::from_str(&format!("{}:{}:{}", key, datatype, data)).unwrap()
}

impl Attributes {
    pub fn get_as_h5(&self) -> AttributesForH5 {
        AttributesForH5 {
            batch_mode:         key_type_data("batch_mode".to_string(),"bool".to_string(), self.batch_mode.to_string()),
            best_search_time:   key_type_data("best_search_time".to_string(),"float".to_string(), self.best_search_time.to_string()),
            candidates:         key_type_data("candidates".to_string(),"float".to_string(), self.candidates.to_string()),
            expect_extra:       key_type_data("expect_extra".to_string(),"bool".to_string(), self.expect_extra.to_string()),
            name:               key_type_data("name".to_string(),"str".to_string(), self.name.to_string()),
            run_count:          key_type_data("run_count".to_string(),"int".to_string(), self.run_count.to_string()),
            distance:           key_type_data("distance".to_string(),"str".to_string(), self.distance.to_string()),
            count:              key_type_data("count".to_string(),"int".to_string(), self.count.to_string()),
            build_time:         key_type_data("build_time".to_string(),"float".to_string(), self.build_time.to_string()),
            index_size:         key_type_data("index_size".to_string(),"float".to_string(), self.index_size.to_string()),
            algo:               key_type_data("algo".to_string(),"str".to_string(), self.algo.to_string()),
            dataset:            key_type_data("dataset".to_string(),"str".to_string(), self.dataset.to_string()),
        }
    }
}

#[derive(Debug, Clone)]
struct ResultFilename {
    path: String,
    name: String,
    filetype: String
}

#[allow(dead_code)]
impl ResultFilename {
    fn filename(&self) -> String {
        format!("{}{}", self.name, self.filetype)
    }

    fn path_and_filename(&self) -> String {
        format!("{}{}{}", self.path, self.name, self.filetype)
    }
}

fn get_result_filename(path: &str, attrs: &Attributes) -> ResultFilename {
    let path = format!("{}/{}/{}/{}/", path, attrs.dataset, attrs.count, attrs.algo);
    ResultFilename {path: path, name: attrs.name.to_string(), filetype: ".hdf5".to_string()}
}

pub fn store_results(results: Vec<(f64, Vec<(usize, f64)>)>, attrs: Attributes) -> hdf5::Result<String> {
    let file = &get_result_filename("results", &attrs);
    // println!("Storing result data into: {}", format!("{}{}{}", file.path, file.name, file.filetype));

    {
        fs::create_dir_all(&file.path).ok();
    }

    let _e = hdf5::silence_errors();
    let hdf5_file = hdf5::File::create(file.path_and_filename());
    match hdf5_file {
        Ok(f) => {
                    let attributes = f.new_dataset::<AttributesForH5>().create("attributes", 1)?;
                    attributes.write(&[attrs.get_as_h5()]).ok();
                    
                    let times = f.new_dataset::<f64>().create("times", results.len())?;
                    let result_count = attrs.count as usize;
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
                    return Ok(file.path_and_filename());
        },
        Err(e) =>   { 
                        println!("Error {}", e);
                        return Err(e);
                    }
    }
    
}