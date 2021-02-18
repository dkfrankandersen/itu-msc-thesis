use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct DataEntry {
    pub index: usize,
    pub distance: f64,
}

impl Eq for DataEntry {}

impl PartialOrd for DataEntry {
    fn partial_cmp(&self, other: &DataEntry) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for DataEntry {
    fn cmp(&self, other: &DataEntry) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}


pub fn test_pq() {
    let mut heap = BinaryHeap::new();
    heap.push(DataEntry {index: 2,  distance: 0.00000002});
    heap.push(DataEntry {index: 7,  distance: 0.00000007});
    heap.push(DataEntry {index: 4,  distance: 0.00000004});
    heap.push(DataEntry {index: 5,  distance: 0.00000005});
    heap.push(DataEntry {index: 6,  distance: 0.00000006});
    heap.push(DataEntry {index: 1,  distance: 0.00000001});
    heap.push(DataEntry {index: 9,  distance: 0.00000009});
    for _ in 0..heap.len() {
        match heap.pop() {
            Some(x) => println!("Some {:?}", x),
            None => println!("None")
        }
    }
}