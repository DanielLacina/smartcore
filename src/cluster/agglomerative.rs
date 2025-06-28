use core::f64;
use std::collections::HashMap;
use std::iter::zip;
use std::marker::PhantomData;

use crate::linalg::basic::arrays::{Array2, ArrayView1};
use crate::numbers::floatnum::FloatNumber;
use crate::numbers::realnum::RealNumber;


pub struct KdNode {
    left: Option<Box<KdNode>>,
    right: Option<Box<KdNode>>,
    label: usize,
    row: Vec<f64>
}

impl KdNode {
    pub fn new(row: Vec<f64>, label: usize) -> Self {
        Self {
            left: None,
            right: None,
            row,
            label
        }

    }
}

pub struct KdTree {
    dim: usize,
    root: Option<Box<KdNode>>
}

impl KdTree {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            root: None
        }
    }

    pub fn create_tree(&mut self, data: Vec<Vec<f64>>, labels: Vec<usize>) {
        let data: Vec<(Vec<f64>, usize)> = zip(data, labels).map(|(row, label)| (row, label)).collect();
        self.root = self._create_tree(data, 0); 

    }

    fn _create_tree(&self, data: Vec<(Vec<f64>, usize)>, depth: usize) -> Option<Box<KdNode>> {
        let i = depth % self.dim;  
        if data.len() == 0 {
            return None;
        }
        let mut data = data;
        data.sort_by(|a, b| a.0[i].partial_cmp(&b.0[i]).unwrap());
        let median = data.len()/2;
        let (row, label) = data[median].clone();
        let mut root = KdNode::new(row, label);
        root.left = self._create_tree(data[0..median].to_vec(), depth + 1);
        root.right = self._create_tree(data[median + 1..data.len()].to_vec(), depth + 1);
        Some(Box::new(root))
    } 

    pub fn nearest(&self, data: &Vec<f64>, label:  usize) -> (f64, usize) {
        let root = self.root.as_ref().unwrap();
        let (min_distance, label) = self._nearest(root, data, label, 0);
        (f64::sqrt(min_distance), label.unwrap())
    }

    fn _nearest(&self, node: &Box<KdNode>, row: &Vec<f64>, label: usize, depth: usize) -> (f64, Option<usize>) {
        let i = depth % self.dim;
        let (primary_node, secondary_node) = if row[i] <= node.row[i] {
           (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };
        let mut min_label = None;
        let mut min_distance = f64::INFINITY;
        if let Some(primary_node) = primary_node {
             (min_distance, min_label) = self._nearest(primary_node, row, label, depth + 1);
        }
        let perp_distance = f64::abs(node.row[i] - row[i]);
        if perp_distance < min_distance {
             if node.label != label {
                let distance_to_node: f64 = zip(node.row.iter(), row.iter()).map(|(v, x)| (v - x).powf(2.0)).sum();
                if distance_to_node < min_distance {
                    min_distance = distance_to_node;
                    min_label = Some(node.label); 
                }
            }
            if let Some(secondary_node) = secondary_node {
                let (secondary_distance, secondary_label) = self._nearest(secondary_node, row, label, depth + 1); 
                if secondary_distance < min_distance {
                    min_distance = secondary_distance;
                    min_label = secondary_label;
                }
            }
        }
        (min_distance, min_label)

    }
    
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