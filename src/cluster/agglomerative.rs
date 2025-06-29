use core::f64;
use std::collections::HashMap;
use std::env::current_dir;
use std::iter::zip;
use std::marker::PhantomData;
use std::usize;

use crate::linalg::basic::arrays::{Array2, ArrayView1};
use crate::numbers::floatnum::FloatNumber;
use crate::numbers::realnum::RealNumber;
use std::cell::RefCell;
use std::rc::{Rc, Weak};

#[derive(Debug, Clone)]
pub enum Direction {
    Left,
    Right,
}

#[derive(Debug)]
pub struct KdNode {
    left: Option<Rc<RefCell<KdNode>>>,
    right: Option<Rc<RefCell<KdNode>>>,
    parent: Weak<RefCell<KdNode>>,
    direction_of_parent: Direction,
    label: usize,
    depth: usize,
    row: Vec<f64>,
}

impl KdNode {
    pub fn new(
        row: Vec<f64>,
        label: usize,
        depth: usize,
        parent: Option<Rc<RefCell<KdNode>>>,
        direction_of_parent: Direction,
    ) -> Self {
        let parent = if let Some(parent) = parent {
            Rc::downgrade(&parent)
        } else {
            Weak::new()
        };
        Self {
            left: None,
            right: None,
            direction_of_parent,
            row,
            label,
            depth,
            parent,
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

    pub fn size(&self) -> usize {
        self.size
    }

   pub fn insert(&mut self, row: Vec<f64>, label: usize) {
    self.size += 1;

    // --- Case 1: The tree is empty. ---
    if self.root.is_none() {
        // Create the root node with the correct constructor
        let root_node = Rc::new(RefCell::new(KdNode::new(
            row,
            label,
            0, // Root depth is 0
            None, // Root has no parent
            Direction::Left, // Direction is arbitrary for the root
        )));
        self.root = Some(root_node);
        return;
    }

    // --- Case 2: The tree is not empty. Use a loop to traverse. ---
    let mut current_parent_rc = self.root.as_ref().unwrap().clone();

    loop {
        let parent_depth;
        let parent_value;
        let split_index;

        // Scope the immutable borrow to be as short as possible
        {
            let parent_ref = current_parent_rc.borrow();
            parent_depth = parent_ref.depth;
            split_index = parent_depth % self.dim;
            parent_value = parent_ref.row[split_index];
        } // `parent_ref` is dropped here, releasing the borrow

        let new_value = row[split_index];
        let go_left = new_value < parent_value;

        // Check if the appropriate child node exists
        let next_node_opt = if go_left {
            current_parent_rc.borrow().left.clone()
        } else {
            current_parent_rc.borrow().right.clone()
        };

        if let Some(next_node_rc) = next_node_opt {
            // A child exists in this direction, continue traversal
            current_parent_rc = next_node_rc;
        } else {
            // No child exists. We found the insertion spot.
            let new_node_depth = parent_depth + 1; // Correct depth calculation!

            // Create the new node, passing a WEAK pointer to the parent
            let new_node = Rc::new(RefCell::new(KdNode::new(
                row,
                label,
                new_node_depth,
                Some(current_parent_rc.clone()), // The crucial change!
                if go_left { Direction::Left } else { Direction::Right },
            )));

            // Now get a mutable borrow on the parent to attach the new child
            if go_left {
                current_parent_rc.borrow_mut().left = Some(new_node);
            } else {
                current_parent_rc.borrow_mut().right = Some(new_node);
            }

            break; // Insertion is complete
        }
    }
} 

    pub fn create(&mut self, data: Vec<Vec<f64>>, labels: Vec<usize>) {
        // Collect into a mutable vector. The data will be partitioned in-place.
        let mut data: Vec<(Vec<f64>, usize)> =
            zip(data, labels).map(|(row, label)| (row, label)).collect();

        // Pass a mutable slice to the recursive helper.
        self.root = self.create_tree(&mut data, 0, None, Direction::Left);
    }

    fn create_tree(
        &mut self,
        data: &mut [(Vec<f64>, usize)],
        depth: usize,
        parent: Option<Rc<RefCell<KdNode>>>,
        direction_of_parent: Direction,
    ) -> Option<Rc<RefCell<KdNode>>> {
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
        let node = Rc::new(RefCell::new(KdNode::new(
            row,
            label,
            depth,
            parent,
            direction_of_parent,
        )));
        {
            // 3. Recurse on the left and right sub-slices. No `to_vec()` needed.
            let mut node_mut = node.borrow_mut();
            node_mut.left =
                self.create_tree(left_data, depth + 1, Some(node.clone()), Direction::Left);
            node_mut.right =
                self.create_tree(right_data, depth + 1, Some(node.clone()), Direction::Right);
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
            &mut min_node,
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
        min_node: &mut Option<Rc<RefCell<KdNode>>>,
    ) {
        // 1. Determine which branch is primary (closer to the query point)
        //    and which is secondary.
        let parent_node = node.borrow();
        let i = parent_node.depth % self.dim;
        let (primary_child, secondary_child) = {
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
            self.search_recursive(child, row, label, min_distance_sq, min_label, min_node);
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
                self.search_recursive(child, row, label, min_distance_sq, min_label, min_node);
            }
        }
    }

    ///
    /// This function handles three cases:
    /// 1. The node is a leaf: It is detached from its parent.
    /// 2. The node is the root and also a leaf: The tree becomes empty.
    /// 3. The node is an internal node: Its data is replaced by the data of a suitable
    ///    successor node (the one with the minimum value in one of its subtrees),
    ///    and then the successor node is recursively removed.
    
    pub fn remove(&mut self, node: Rc<RefCell<KdNode>>)  {
        self.remove_node(node);
        self.size -= 1;
    }   
    fn remove_node(&mut self, node: Rc<RefCell<KdNode>>) {
        // Determine if the node is an internal node by finding a replacement.
        // A leaf node will not have a replacement.
        let min_node_opt = {
            let node_ref = node.borrow();
            // If the node has no children, it's a leaf, so no replacement search is needed.
            if node_ref.left.is_none() && node_ref.right.is_none() {
                None
            } else {
                let split_index = node_ref.depth % self.dim;
                let mut min_value = f64::INFINITY;
                let mut min_node = None;

                // Find the minimum-valued node in the subtrees to act as a replacement.
                // The standard algorithm often prefers the right subtree, but searching both is valid.
                if let Some(right_node) = node_ref.right.clone() {
                    self.find_min(&mut min_node, &mut min_value, right_node, split_index);
                }
                if let Some(left_node) = node_ref.left.clone() {
                    self.find_min(&mut min_node, &mut min_value, left_node, split_index);
                }
                min_node
            }
        };

        if let Some(min_node) = min_node_opt {
            // --- Case 1: The node is an internal node. ---
            // We replace this node's data with the replacement's data, then remove the replacement.

            // Scope the borrows tightly to avoid conflicts.
            {
                let min_node_ref = min_node.borrow();
                let mut node_mut = node.borrow_mut();
                // Copy data from the replacement node to the node we want to "remove".
                node_mut.label = min_node_ref.label.clone();
                node_mut.row = min_node_ref.row.clone();
            } // All borrows are released here.

            // Now, recursively remove the original `min_node`, which is now redundant.
            // This is safe because no borrows are held across the recursive call.
            self.remove_node(min_node);
        } else {
            // --- Case 2: The node is a leaf. ---
            // We detach it from its parent.

            // Get the parent weak pointer and upgrade it to an Rc.
            let parent_opt = node.borrow().parent.upgrade();

            if let Some(parent_rc) = parent_opt {
                // The node has a parent, so detach it.
                let mut parent_mut = parent_rc.borrow_mut();
                // Determine which child to remove (left or right).
                let direction = node.borrow().direction_of_parent.clone();
                match direction {
                    Direction::Left => parent_mut.left = None,
                    Direction::Right => parent_mut.right = None,
                }
            } else {
                // The node has no parent, so it must be the root.
                // Since it's also a leaf, deleting it means the tree is now empty.
                self.root = None;
            }
        }
    }

    /// Recursively finds the node with the minimum value along a given dimension (`split_index`)
    /// within a subtree rooted at `node`.
    pub fn find_min(
        &self,
        min_node: &mut Option<Rc<RefCell<KdNode>>>,
        min_value: &mut f64,
        node: Rc<RefCell<KdNode>>,
        split_index: usize,
    ) {
        let node_ref = node.borrow();

        // Check the current node's value.
        let parent_node_value = node_ref.row[split_index];
        if parent_node_value < *min_value {
            *min_value = parent_node_value;
            *min_node = Some(node.clone());
        }

        // To find the true minimum, we must search the entire subtree unconditionally.
        if let Some(left_node) = node_ref.left.clone() {
            self.find_min(min_node, min_value, left_node, split_index);
        }
        if let Some(right_node) = node_ref.right.clone() {
            self.find_min(min_node, min_value, right_node, split_index);
        }
    }
}

// The Cluster struct no longer needs PartialEq as we will use IDs for comparison.
// Clone is kept for convenience, but the main loop avoids using it.
#[derive(Clone, Debug)]
pub struct Cluster {
    sum: Vec<f64>,
    average: Vec<f64>,
    /// Contains the indices of the original data points belonging to this cluster.
    values: Vec<usize>,
}

impl Cluster {
    /// Creates a new cluster from a single data point.
    pub fn new(point_index: usize, data: Vec<f64>) -> Self {
        Self {
            sum: data.clone(),
            average: data,
            values: vec![point_index],
        }
    }

    /// Merges another cluster into this one efficiently.
    /// This method now takes ownership of `other` to avoid cloning its internal vectors.
    /// It also performs calculations in-place to prevent new memory allocations.
    pub fn add_cluster(&mut self, mut other: Cluster) {
        // Extend the list of point indices. append is O(1).
        self.values.append(&mut other.values);

        // Update the sum of all points' coordinates in-place.
        for (s_val, o_val) in self.sum.iter_mut().zip(other.sum.iter()) {
            *s_val += *o_val;
        }

        // Recalculate the average (centroid) in-place.
        let n = self.values.len() as f64;
        for i in 0..self.average.len() {
            self.average[i] = self.sum[i] / n;
        }
    }
}

pub struct AgglomerativeClustering<TX, X> {
    pub labels: Vec<usize>,
    _phantom_tx: PhantomData<TX>,
    _phantom_x: PhantomData<X>,
}

impl<TX: FloatNumber + RealNumber, X: Array2<TX>> AgglomerativeClustering<TX, X>
{
   pub fn fit(data: &X, n_clusters: usize) -> Result<Vec<usize>, String> {
    let (num_samples, num_features) = data.shape();
    let data: Vec<Vec<f64>> = (0..num_samples).map(|i| data.get_row(i).iterator(0).map(|x| x.to_f64().unwrap()).collect::<Vec<f64>>()).collect();
    if num_samples < n_clusters {
        return Err("Number of samples must be greater than or equal to n_clusters.".to_string());
    }

    // --- 1. Initialization ---
    let mut kdtree = KdTree::new(num_features);
    kdtree.create(data.clone(), (0..num_samples).collect());
    let mut clusters = HashMap::with_capacity(num_samples);

    for (i, row) in data.into_iter().enumerate() {
        // Each point starts as its own cluster. The cluster ID is the point's original index.
        let cluster = Cluster::new(i, row.clone());
        clusters.insert(i, cluster);
    }

    // --- 2. Main Clustering Loop ---
    // This loop implements the reciprocal nearest neighbors strategy.

    // Start with an arbitrary cluster (id 0) and find its nearest neighbor.
    let start_index = 0;
    let (_, mut a_index, mut a_node) = kdtree.nearest(&clusters[&start_index].average, start_index);
    let (_, mut b_index, mut b_node) = kdtree.nearest(&clusters[&a_index].average, a_index);

    while clusters.len() > n_clusters {
        let b_average = clusters[&b_index].average.clone();
        let (_, c_index, c_node) = kdtree.nearest(&b_average, b_index);
        // --- Reciprocal Match Check ---
        if a_index == c_index {
            kdtree.remove(a_node.clone());
            kdtree.remove(b_node.clone());
            let b_cluster = clusters.remove(&b_index).unwrap();
            let a_cluster = clusters.get_mut(&a_index).unwrap();
            a_cluster.add_cluster(b_cluster);
            kdtree.insert(a_cluster.average.clone(), a_index);
            let (_, b_index_, b_node_) = kdtree.nearest(&clusters[&a_index].average, a_index);
            b_index = b_index_;
            b_node = b_node_;
        } else {
            a_index = b_index;
            a_node = b_node;
            b_index = c_index;
            b_node = c_node;
        }
    }

    let mut labels = vec![0; num_samples];
    for (final_label, (_, cluster)) in clusters.iter().enumerate() {
        for original_point_index in cluster.values.iter() {
            labels[*original_point_index] = final_label;
        }
    }

    Ok(labels)
} 
}
