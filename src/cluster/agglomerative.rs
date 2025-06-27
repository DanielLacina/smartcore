use std::collections::BinaryHeap;
use std::marker::PhantomData;
use std::cmp::Ordering;

use crate::linalg::basic::arrays::Array2;
use crate::numbers::floatnum::FloatNumber;
use crate::numbers::realnum::RealNumber;

// --- Helper Structs ---

// Represents the distance between two nodes (points or clusters).
#[derive(Debug, Clone, PartialEq)]
pub struct PairwiseDistance {
    pub node1: usize,
    pub node2: usize,
    pub distance: f64,
}

// We need to implement Eq, Ord, and PartialOrd to use this in a BinaryHeap.
// The default BinaryHeap is a max-heap, so we reverse the ordering
// to make it behave like a min-heap (smallest distance has highest priority).
impl Eq for PairwiseDistance {}

impl PartialOrd for PairwiseDistance {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.distance.partial_cmp(&self.distance)
    }
}

impl Ord for PairwiseDistance {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}


// Represents a node in the final dendrogram (the output hierarchy).
#[derive(Debug, Clone)]
pub struct LinkageNode {
    pub left: Option<Box<LinkageNode>>,
    pub right: Option<Box<LinkageNode>>,
    pub index: usize, // For leaf nodes, this is the original point index.
    pub distance: f64, // The distance at which this node/merge was created.
}

impl LinkageNode {
    /// Creates a new leaf node for an original data point.
    fn new_leaf(index: usize) -> Self {
        Self { left: None, right: None, index, distance: 0.0 }
    }

    /// Creates a new internal node representing a merge.
    fn new_internal(left: LinkageNode, right: LinkageNode, distance: f64) -> Self {
        Self {
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            index: usize::MAX, // Sentinel value indicating it's not a leaf.
            distance,
        }
    }
}


// --- The Efficient Data Structure: Disjoint Set Union (DSU) ---

/// A Disjoint Set Union (DSU) data structure, also known as Union-Find.
/// It tracks a collection of disjoint sets and provides two main operations:
/// 1. `find`: Determine which set an element belongs to (i.e., find its root).
/// 2. `union`: Join two sets together.
/// This implementation uses path compression and union by size for optimal performance.
pub struct Dsu {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl Dsu {
    /// Creates a new DSU with `n` elements, each in its own set.
    pub fn new(n: usize) -> Self {
        Dsu {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }

    /// Finds the representative (or root) of the set containing element `i`.
    /// Implements path compression for efficiency.
    pub fn find(&mut self, mut i: usize) -> usize {
        let mut root = i;
        while root != self.parent[root] {
            root = self.parent[root];
        }
        // Path compression: set parent of all nodes on the path to the root
        while i != root {
            let next_i = self.parent[i];
            self.parent[i] = root;
            i = next_i;
        }
        root
    }

    /// Merges the sets containing elements `i` and `j`.
    /// Returns the root of the new merged set.
    pub fn union(&mut self, i: usize, j: usize) -> usize {
        let mut root_i = self.find(i);
        let mut root_j = self.find(j);
        if root_i != root_j {
            // Union by size: merge smaller tree into larger tree
            if self.size[root_i] < self.size[root_j] {
                std::mem::swap(&mut root_i, &mut root_j);
            }
            self.parent[root_j] = root_i;
            self.size[root_i] += self.size[root_j];
        }
        root_i
    }
}


// --- Main Clustering Struct and Refactored `fit` Function ---

pub struct AgglomerativeClustering<TX, X> {
    pub labels: Vec<usize>,
    pub dendrogram: LinkageNode,
    _phantom_tx: PhantomData<TX>,
    _phantom_x: PhantomData<X>,
}

impl<TX: FloatNumber + RealNumber, X: Array2<TX>> AgglomerativeClustering<TX, X>
where
    TX: Copy + Into<f64> + std::ops::Sub<Output = TX>,
    f64: From<TX>,
{
    /// An efficient implementation of agglomerative clustering using a
    /// priority queue (min-heap) and a Disjoint Set Union (DSU) data structure.
    pub fn fit(data: &X) -> Result<Self, String> {
        let (num_samples, _) = data.shape();
        if num_samples == 0 {
            return Err("Cannot cluster empty data.".to_string());
        }
        if num_samples == 1 {
            return Ok(Self {
                labels: vec![0],
                dendrogram: LinkageNode::new_leaf(0),
                _phantom_tx: PhantomData,
                _phantom_x: PhantomData,
            });
        }

        // 1. Calculate all pairwise distances and populate a min-priority-queue.
        let mut pq = BinaryHeap::new();
        for i in 0..num_samples {
            let row_i = data.get_row(i);
            for j in (i + 1)..num_samples {
                let row_j = data.get_row(j);

                // Efficient distance calculation using iterators (no allocations)
                let distance: f64 = row_i.iterator(0).zip(row_j.iterator(0)).map(|(&a, &b)| {
                    let diff = f64::from(a) - f64::from(b);
                    diff * diff
                }).sum();

                pq.push(PairwiseDistance { node1: i, node2: j, distance });
            }
        }

        // 2. Initialize DSU for tracking cluster membership and storage for dendrogram nodes.
        let mut dsu = Dsu::new(num_samples);
        let mut nodes: Vec<Option<LinkageNode>> = (0..num_samples)
            .map(|i| Some(LinkageNode::new_leaf(i)))
            .collect();

        // 3. Main merging loop. We need to perform n-1 merges.
        let required_merges = num_samples - 1;
        for _ in 0..required_merges {
            // 4. Pop the next closest pair from the priority queue until we find a valid merge.
            loop {
                let closest_pair = pq.pop().ok_or("Priority queue exhausted before all merges were complete.")?;

                let root1 = dsu.find(closest_pair.node1);
                let root2 = dsu.find(closest_pair.node2);

                // If they are not already in the same cluster, merge them.
                if root1 != root2 {
                    // Take ownership of the nodes to be merged.
                    let node1 = nodes[root1].take().expect("Node 1 should exist.");
                    let node2 = nodes[root2].take().expect("Node 2 should exist.");

                    // Create the new parent node representing the merge.
                    let new_node = LinkageNode::new_internal(node1, node2, closest_pair.distance.sqrt());

                    // Merge the sets and get the new root for the combined cluster.
                    let new_root = dsu.union(root1, root2);

                    // Store the new merged node at the root's index.
                    nodes[new_root] = Some(new_node);

                    // Break the inner loop since we've completed a merge.
                    break;
                }
                // If they were already in the same cluster, this pair is "stale".
                // The loop continues, popping the next closest pair.
            }
        }

        // 5. Extract the final dendrogram, which is the last remaining node.
        let final_root = dsu.find(0);
        let dendrogram = nodes[final_root].take().expect("Final dendrogram should exist.");

        Ok(Self {
            labels: Vec::new(), // Label assignment can be a separate step.
            dendrogram,
            _phantom_tx: PhantomData,
            _phantom_x: PhantomData,
        })
    }
}
