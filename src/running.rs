use crate::util::{hdf5_store_file, RunParameters, store_results_and_fix_attributes};

pub fn compute_timing_and_store(best_search_time: f64, build_time: f64, results: Vec::<(f64, Vec<(usize, f64)>)>, 
                                            results_per_query: usize, dataset_size: usize, parameters: RunParameters) {
    let mut total_time: f64 = 0.;
    let mut total_candidates: usize = 0;
    for (time, candidates) in results.iter() {
        total_time += time;
        total_candidates += candidates.len();
    }

    let search_time = total_time / dataset_size as f64;
    let avg_candidates = total_candidates as f64 / dataset_size as f64;
    let best_search_time = { if best_search_time < search_time { best_search_time } else { search_time }};

    let attrs = hdf5_store_file::Attributes {
        build_time,
        index_size: 0.,
        algo: parameters.algorithm.clone(),
        dataset: parameters.dataset.clone(),

        batch_mode: false,
        best_search_time,
        candidates: avg_candidates,
        count: results_per_query as u32,
        distance: parameters.metric.clone(),
        expect_extra: false,
        name: parameters.algo_definition(),
        run_count: 1
    };
    // println!("Store results into HD5F file");
    store_results_and_fix_attributes(results, attrs);
}