use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

pub fn run(file: String) -> Result<(), ()> {
    Python::with_gil(|py| {
        main_(py, file).map_err(|e| {
          // We can't display Python exceptions via std::fmt::Display,
          // so print the error here manually.
          e.print_and_set_sys_last_vars(py);
        })
    })
}

fn main_(py: Python, file2: String) -> PyResult<()> {
    let locals = [("os", py.import("os")?)].into_py_dict(py);
    let code = format!("os.system('python3 src/util/hdf5_attributes_fix_single.py {}')", file2);
    py.eval(&code, None, Some(&locals))?;
    Ok(())
}