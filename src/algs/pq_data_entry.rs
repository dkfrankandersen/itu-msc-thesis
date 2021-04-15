use std::cmp::Ordering;

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct PQDataEntry {
    pub cluster: usize,
    pub centroid: usize,
    pub distance: f64,
}

impl Eq for PQDataEntry {}

impl PartialOrd for PQDataEntry {
    fn partial_cmp(&self, other: &PQDataEntry) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for PQDataEntry {
    fn cmp(&self, other: &PQDataEntry) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

// #[cfg(test)]
// mod tests {
//     #[test]
//     fn given_2_3_1_return_3() {
//         use std::collections::BinaryHeap;
//         use crate::algs::pq_data_entry::{DataEntry};

//         let mut heap = BinaryHeap::new();
//         heap.push(DataEntry {index: 2,  distance: 0.2});
//         heap.push(DataEntry {index: 3,  distance: 0.3});
//         heap.push(DataEntry {index: 1,  distance: 0.1});
//         let idx = (Some(heap.pop()).unwrap()).unwrap();

//         assert_eq!((idx.index, idx.distance), (3, 0.3));
//     }
//     #[test]
//     fn given_neg_2_3_1_return_1() {
//         use std::collections::BinaryHeap;
//         use crate::algs::pq_data_entry::{DataEntry};

//         let mut heap = BinaryHeap::new();
//         heap.push(DataEntry {index: 2,  distance: -0.2});
//         heap.push(DataEntry {index: 3,  distance: -0.3});
//         heap.push(DataEntry {index: 1,  distance: -0.1});
//         let idx = (Some(heap.pop()).unwrap()).unwrap();

//         assert_eq!((idx.index, idx.distance), (1, -0.1));
//     }
// }