use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::cmp::Reverse;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct DataEntry {
    index: usize,
    distance: f64,
}

impl Eq for DataEntry {}

impl PartialOrd for DataEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.distance.partial_cmp(&self.distance)
    }
}

impl Ord for DataEntry {
    fn cmp(&self, other: &DataEntry) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl PartialEq for DataEntry {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }

pub fn test_pq() {
    let mut heap = BinaryHeap::new();

    heap.push(DataEntry {index: 2,  distance: 0.00000001});
    heap.push(DataEntry {index: 7,  distance: 0.00000001});
    heap.push(DataEntry {index: 4,  distance: 0.00000001});
    heap.push(DataEntry {index: 5,  distance: 1.00000005});
    heap.push(DataEntry {index: 6,  distance: 0.00000006});
    heap.push(DataEntry {index: 1,  distance: 0.00000001});
    heap.push(DataEntry {index: 9,  distance: 0.00000009});
    heap.push(DataEntry {index: 10,  distance: 0.1000000001});
    heap.push(DataEntry {index: 11,  distance: 0.800000009});
    
    for _ in 0..heap.len() {
        match heap.pop() {
            Some(x) => println!("Some {:?}", x),
            None => println!("None")
        }
    }
    

}