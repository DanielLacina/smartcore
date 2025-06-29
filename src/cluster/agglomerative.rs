use core::f64;
use std::collections::HashMap;
use std::env::current_dir;
use std::iter::zip;
use std::marker::PhantomData;
use std::usize;

use crate::linalg::basic::arrays::{Array2, ArrayView1};
use crate::numbers::floatnum::FloatNumber;
use crate::numbers::realnum::RealNumber;
use std::rc::{Rc, Weak};
use std::cell::RefCell;

#[derive(Debug)]
pub struct KdNode {
    left: Option<Rc<RefCell<KdNode>>>,
    right: Option<Rc<RefCell<KdNode>>>,
    parent: Weak<RefCell<KdNode>>,
    label: usize,
    depth: usize,
    row: Vec<f64>,
}

impl KdNode {
    pub fn new(row: Vec<f64>, label: usize, depth: usize, parent: Option<Rc<RefCell<KdNode>>>) -> Self {
        let parent = if let Some(parent) = parent {
            Rc::downgrade(&parent)
        }  else {
            Weak::new()
        };
        Self {
            left: None,
            right: None,
            row,
            label,
            depth,
            parent
        }
    }
}

pub struct KdTree {
    dim: usize,
    root: Option<Rc<RefCell<KdNode>>>,
    size: usize,
}

impl KdTree {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            root: None,
            size: 0,
        }
    }

    pub fn size(&self) -> usize{
        self.size
    }

    pub fn create_tree(&mut self, data: Vec<Vec<f64>>, labels: Vec<usize>) {
        // Collect into a mutable vector. The data will be partitioned in-place.
        let mut data: Vec<(Vec<f64>, usize)> =
            zip(data, labels).map(|(row, label)| (row, label)).collect();

        // Pass a mutable slice to the recursive helper.
        self.root = self._create_tree(&mut data, 0, None);
    }

    fn _create_tree(&mut self, data: &mut [(Vec<f64>, usize)], depth: usize, parent: Option<Rc<RefCell<KdNode>>>) -> Option<Rc<RefCell<KdNode>>> {
        if data.is_empty() {
            return None;
        }

        let axis = depth % self.dim;
        let median_idx = data.len() / 2;

        // 1. Partition the slice in-place to find the median along the current axis.
        //    This is O(n) on average, much faster than a full O(n log n) sort.
        data.select_nth_unstable_by(median_idx, |a, b| {
            a.0[axis].partial_cmp(&b.0[axis]).unwrap()
        });

        // 2. Split the slice into three parts without copying memory:
        //    - The left sub-slice
        //    - The median element itself
        //    - The right sub-slice
        let (left_data, rest) = data.split_at_mut(median_idx);
        let (median_element, right_data) = rest.split_at_mut(1);

        // The median element becomes the root of this sub-tree. We still clone it,
        // as the node needs to own its data. This is a small, necessary copy.
        let (row, label) = median_element[0].clone();
        let node = Rc::new(RefCell::new(KdNode::new(row, label, depth, parent)));
        {
        // 3. Recurse on the left and right sub-slices. No `to_vec()` needed.
            let mut node_mut = node.borrow_mut(); 
            node_mut.left = self._create_tree(left_data, depth + 1, Some(node.clone()));
            node_mut.right = self._create_tree(right_data, depth + 1, Some(node.clone()));
        }

        self.size += 1;
        Some(node)
    }

    pub fn nearest(&mut self, row: &Vec<f64>, label: usize) -> (f64, usize, Rc<RefCell<KdNode>>) {
        // Initialize with the root node's data.
        let mut min_distance_sq = f64::INFINITY; 
        let mut min_label = usize::MAX; 

        let mut min_node = None;

        // Start the recursive search.
        self.search_recursive(
            self.root.clone().unwrap(),
            row,
            label,
            &mut min_distance_sq,
            &mut min_label,
            &mut min_node
        );

        (f64::sqrt(min_distance_sq), min_label, min_node.unwrap())
    }

    /// Private recursive helper for the nearest neighbor search.
    fn search_recursive(
        &self,
        node: Rc<RefCell<KdNode>>,
        row: &Vec<f64>,
        label: usize,
        min_distance_sq: &mut f64,
        min_label: &mut usize,
        min_node: &mut Option<Rc<RefCell<KdNode>>>
    ) {
        // 1. Determine which branch is primary (closer to the query point)
        //    and which is secondary.
        let parent_node = node.borrow();
        let i = parent_node.depth % self.dim;
        let (primary_child, secondary_child) =
        {
            let left_node = parent_node.left.clone();   
            let right_node = parent_node.right.clone();
            if row[i] <= parent_node.row[i] {
                (left_node, right_node)
            } else {
                (right_node, left_node)
            }
        };

        // 2. Recurse down the primary branch first.
        //    This will explore the most promising path to the bottom of the tree.
        if let Some(child) = primary_child {
            self.search_recursive(
                child,
                row,
                label,
                min_distance_sq,
                min_label,
                min_node
            );
        }

        // 4. Check the secondary branch.
        //    We only explore this branch if it's possible it could contain a
        //    closer point. This is the core optimization of the k-d tree.
        let perp_distance_sq = (parent_node.row[i] - row[i]).powi(2);

        // This is the "ball within bounds" check. If the squared perpendicular
        // distance to the splitting plane is less than our current best squared
        // distance, the hypersphere around our query point intersects the plane,
        // so we must explore the other side.
        if perp_distance_sq < *min_distance_sq {
            if label != parent_node.label {
                let distance_sq: f64 = zip(parent_node.row.iter(), row.iter())
                    .map(|(v, x)| (v - x).powi(2))
                    .sum();

                if distance_sq < *min_distance_sq {
                    *min_distance_sq = distance_sq;
                    *min_label = parent_node.label;
                    *min_node = Some(node.clone());
                }
            }
            if let Some(child) = secondary_child {
                self.search_recursive(
                    child,
                    row,
                    label,
                    min_distance_sq,
                    min_label,
                    min_node
                );
            }
        }
    }

    // pub fn delete(&mut self, node: Rc<RefCell<KdNode>>) {
    //     let depth = node.borrow().depth % self.dim;
    //     let mut min_value = f64::INFINITY;
    //     let mut min_node = None;
    //     self.find_min(min_node, &mut min_value, node, split_index);
    //     if self.dim > 1 {
            
    //     }
    // }

    // pub fn find_min(&self, min_node: &mut Option<Rc<RefCell<KdNode>>>, min_value: &mut f64, node: Rc<RefCell<KdNode>>, split_index: usize) {
    //     let parent_node = node.borrow();
    //     let cur_split_index = parent_node.depth % self.dim;
    //     if let Some(left_node) = parent_node.left.clone() {
    //         self.find_min(min_node, min_value, left_node, split_index);
    //     }
    //     if cur_split_index != split_index {
    //         if let Some(right_node) = parent_node.right.clone() {
    //             self.find_min(min_node, min_value, right_node, split_index);
    //         }
    //     }
    //     let parent_node_value = parent_node.row[split_index];
    //     if parent_node_value < *min_value {
    //         *min_value = parent_node_value;
    //         *min_node = Some(node.clone());
    //     }
    // }
}

// // The Cluster struct no longer needs PartialEq as we will use IDs for comparison.
// // Clone is kept for convenience, but the main loop avoids using it.
// #[derive(Clone, Debug)]
// pub struct Cluster {
//     sum: Vec<f64>,
//     average: Vec<f64>,
//     /// Contains the indices of the original data points belonging to this cluster.
//     values: Vec<usize>,
// }

// impl Cluster {
//     /// Creates a new cluster from a single data point.
//     pub fn new(point_index: usize, data: Vec<f64>) -> Self {
//         Self {
//             sum: data.clone(),
//             average: data,
//             values: vec![point_index],
//         }
//     }

//     /// Merges another cluster into this one efficiently.
//     /// This method now takes ownership of `other` to avoid cloning its internal vectors.
//     /// It also performs calculations in-place to prevent new memory allocations.
//     pub fn add_cluster(&mut self, mut other: Cluster) {
//         // Extend the list of point indices. append is O(1).
//         self.values.append(&mut other.values);

//         // Update the sum of all points' coordinates in-place.
//         for (s_val, o_val) in self.sum.iter_mut().zip(other.sum.iter()) {
//             *s_val += *o_val;
//         }

//         // Recalculate the average (centroid) in-place.
//         let n = self.values.len() as f64;
//         for i in 0..self.average.len() {
//             self.average[i] = self.sum[i] / n;
//         }
//     }
// }

// pub struct AgglomerativeClustering<TX, X> {
//     pub labels: Vec<usize>,
//     _phantom_tx: PhantomData<TX>,
//     _phantom_x: PhantomData<X>,
// }

// impl<TX: FloatNumber + RealNumber, X: Array2<TX>> AgglomerativeClustering<TX, X>
// where
//     TX: Copy + Into<f64> + std::ops::Sub<Output = TX>,
//     f64: From<TX>,
// {
//     pub fn fit(data: &X, n_clusters: usize) -> Result<Self, String> {
//         let (num_samples, num_features) = data.shape();
//         if num_samples < 2 {
//             return Err("At least 2 samples are required for clustering.".to_string());
//         }

//         let mut kdtree = KdTree::new(num_features);
//         // The HashMap is the single source of truth for all cluster data.
//         let mut clusters = HashMap::with_capacity(num_samples);

//         // --- Initialization ---
//         for i in 0..num_samples {
//             let point_data: Vec<f64> = data.get_row(i).iterator(0).map(|x| x.to_f64().unwrap().collect());

//             // Create the initial cluster.
//             let cluster = Cluster::new(i, point_data.clone());

//             // Add the cluster's location and ID to the spatial index.
//             kdtree.add(point_data, i).unwrap();
//             clusters.insert(i, cluster);
//         }

//         // --- Main Clustering Loop ---

//         // Start with an arbitrary cluster (e.g., id 0) and its nearest neighbor.
//         let mut a_id = 0;
//         let a_average = clusters.get(&a_id).unwrap().average.clone();
//         let b_results = kdtree.nearest(&a_average, 2, &squared_euclidean).unwrap();
//         let mut b_id = *b_results[1].1;
//         let mut clusters_len = clusters.len();

//         while clusters_len > n_clusters {
//             println!("{}, {}", a_id, b_id);
//             // Find C, the nearest neighbor of B.
//             // We clone `b_average` because the kdtree query requires an owned value or a reference.
//             let b_average = clusters.get(&b_id).unwrap().average.clone();
//             let c_results = kdtree.nearest(&b_average, 2, &squared_euclidean).unwrap();
//             let mut c_id = *c_results[1].1;
//             if c_id == b_id {
//                 c_id =  *c_results[0].1;
//             }
//             // Check for a reciprocal best match using efficient ID comparison.
//             if a_id == c_id {
//                 // --- Merge B into A ---

//                 // Store old average of A before it gets modified.
//                 let old_a_average = clusters.get(&a_id).unwrap().average.clone();

//                 // Remove B from the HashMap to take ownership of it.
//                 let b_cluster = clusters.remove(&b_id).unwrap();
//                 clusters_len -= 1;
//                 // Also remove B from the kdtree.
//                 kdtree.remove(&b_cluster.average, &b_id).unwrap();

//                 // Get a mutable reference to A to perform the merge.
//                 let a_cluster_mut = clusters.get_mut(&a_id).unwrap();
//                 // Remove the old A from the kdtree before its average changes.
//                 kdtree.remove(&old_a_average, &a_id).unwrap();

//                 // Perform the efficient, in-place merge. This consumes b_cluster.
//                 a_cluster_mut.add_cluster(b_cluster);

//                 // Add the updated cluster A back to the kdtree with its new average.
//                 kdtree.add(a_cluster_mut.average.clone(), a_id).unwrap();
//                 // For the next iteration, find the new nearest neighbor for our merged cluster.
//                 if clusters_len > n_clusters {
//                     let new_a_average = &a_cluster_mut.average;
//                     let new_b_results = kdtree.nearest(new_a_average, 2, &squared_euclidean).unwrap();
//                     b_id = *new_b_results[1].1;
//                 }
//             } else {
//                 // Not a reciprocal match, so we "walk" to the next pair.
//                 a_id = b_id;
//                 b_id = c_id;
//             }
//             clusters_len = clusters.len();
//         }
//         let mut labels = vec![0; num_samples];

//         for (i, (_, cluster)) in clusters.iter().enumerate() {
//             for index in cluster.values.iter() {
//                  labels[*index] = i;
//             }
//         }

//         // At this point, `clusters` contains the single, final cluster.
//         // Label assignment can be implemented here if needed.
//         Ok(Self {
//             labels,
//             _phantom_tx: PhantomData,
//             _phantom_x: PhantomData,
//         })
//     }
// }
