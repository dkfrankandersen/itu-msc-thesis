extern crate ndarray;
extern crate hdf5;
use std::time::{Instant};
mod dataset;
mod distance;
mod bruteforce;
use priority_queue::PriorityQueue;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(PartialEq, Debug)]
struct MinFloat(f64);

impl Eq for MinFloat {}

impl PartialOrd for MinFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.0.partial_cmp(&self.0)
    }
}

impl Ord for MinFloat {
    fn cmp(&self, other: &MinFloat) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}


fn main() {

    let mut pq = PriorityQueue::<usize, MinFloat>::new();

    pq.push(1, MinFloat(0.00000000000000001));
    pq.push(7, MinFloat(0.00000000000000007));
    pq.push(4, MinFloat(0.00000000000000004));
    pq.push(2, MinFloat(0.00000000000000002));
    pq.push(2, MinFloat(0.00000000000000002));
    pq.push(9, MinFloat(0.00000000000000009));
    pq.push(3, MinFloat(0.00000000000000003));

    for (item, _) in pq.into_sorted_iter() {
        println!("{}", item);
    }

    let mut minheap = BinaryHeap::new();
    minheap.push(MinFloat(2.0));
    minheap.push(MinFloat(1.0));
    minheap.push(MinFloat(42.0));
    if let Some(MinFloat(root)) = minheap.pop() {
        println!("{:?}", root);
    }
    return;

    let _e = hdf5::silence_errors();
    let _filename = "datasets/glove-100-angular.hdf5";
    let file = &dataset::get_dataset(_filename);
    
    let ds_train = dataset::get_dataset_f64(file, "train");
    let ds_train_norm = dataset::normalize_all(ds_train);
    
    let ds_test = dataset::get_dataset_f64(file, "test");
    let ds_test_norm = &dataset::normalize_all(ds_test);

    let ds_distance = dataset::get_dataset_f64(file, "distances");
    let ds_distance_norm = dataset::normalize_all(ds_distance);

    let ds_neighbors = dataset::get_dataset_i64(file, "neighbors");

    let time_start = Instant::now();

    // linear scan
    println!("bruteforce_search started at {:?}", time_start);
    for (i,p) in ds_test_norm.outer_iter().enumerate() {
        let (best_idx, best_dist) = bruteforce::bruteforce_search(&p, &ds_train_norm.view(), crate::distance::DistType::Cosine);
    }

    let time_finish = Instant::now();
    println!("Duration: {:?}", time_finish.duration_since(time_start));
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn another() {
        panic!("Make this test fail");
    }
}