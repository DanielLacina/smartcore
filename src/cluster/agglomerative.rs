use core::f64;
use std::collections::{HashMap, HashSet};
use std::env::current_dir;
use std::hash::Hash;
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

   pub fn add(&mut self, row: Vec<f64>, label: usize) -> Rc<RefCell<KdNode>> {
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
        self.root = Some(root_node.clone());
        return root_node;
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
            // No child exists. We found the addion spot.
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
                current_parent_rc.borrow_mut().left = Some(new_node.clone());
            } else {
                current_parent_rc.borrow_mut().right = Some(new_node.clone());
            }
            return new_node
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
    
    // In your `impl KdTree`:

/// Removes a node from the tree and reports if any tracked data was moved.
///
/// # Arguments
/// * `node_to_remove`: An `Rc` pointing to the node that should be removed.
/// * `tracked_labels`: A set of labels the caller is "listening" for. If the data
///   from a tracked label is swapped into a different node during removal, this
///   function reports the change.
///
/// # Returns
/// A `HashMap` mapping a tracked label to the `Rc` of the node that now holds its data.
pub fn remove(
    &mut self,
    node_to_remove: Rc<RefCell<KdNode>>,
    tracked_labels: &HashSet<usize>,
) -> HashMap<usize, Rc<RefCell<KdNode>>> {
    // This map will store the new locations of any tracked data that gets moved.
    let mut swapped_locations = HashMap::new();
    
    // Call the recursive helper function to perform the removal.
    self.remove_recursive(node_to_remove, &mut swapped_locations, tracked_labels);
    
    self.size -= 1;
    swapped_locations
}

/// The private, recursive helper for node removal.
///
/// This function implements the "delete by copying" algorithm and handles two main cases:
/// 1.  **Internal Node**: Finds a replacement, copies its data into the target node,
///     and then recursively deletes the now-empty replacement node.
/// 2.  **Leaf Node**: Simply detaches the node from its parent.
fn remove_recursive(
    &mut self,
    node_to_remove: Rc<RefCell<KdNode>>,
    swapped_locations: &mut HashMap<usize, Rc<RefCell<KdNode>>>,
    tracked_labels: &HashSet<usize>,
) {
    // --- Step 1: Find a suitable replacement from a subtree. ---
    // If no replacement is found, it means `node_to_remove` is a leaf.
    let replacement_node_opt = {
        let node_ref = node_to_remove.borrow();
        if node_ref.left.is_none() && node_ref.right.is_none() {
            None
        } else {
            let split_index = node_ref.depth % self.dim;
            let mut min_value = f64::INFINITY;
            let mut min_node = None;

            // Search both subtrees for the best possible replacement.
            if let Some(right_node) = node_ref.right.clone() {
                self.find_min(&mut min_node, &mut min_value, right_node, split_index);
            }
            if let Some(left_node) = node_ref.left.clone() {
                self.find_min(&mut min_node, &mut min_value, left_node, split_index);
            }
            min_node
        }
    };

    if let Some(replacement_node) = replacement_node_opt {
        // --- Case 1: The node is an INTERNAL NODE. ---
        // Overwrite this node's data with the replacement's, then delete the replacement.
        
        // Use a tightly scoped borrow to perform the data swap safely.
        {
            let replacement_ref = replacement_node.borrow();
            let mut node_to_remove_mut = node_to_remove.borrow_mut();

            // --- Track Data Swaps ---
            // If an external system is tracking the replacement node, we must report
            // that its data has now been moved into the `node_to_remove` location.
            if tracked_labels.contains(&replacement_ref.label) {
                swapped_locations.insert(replacement_ref.label, node_to_remove.clone());
            }

            // Copy the replacement's data, effectively "moving" it up the tree.
            node_to_remove_mut.label = replacement_ref.label;
            node_to_remove_mut.row = replacement_ref.row.clone();
        } // All borrows are released here, making the recursive call safe.

        // Now, recursively call this function to remove the now-redundant
        // replacement node from its original position.
        self.remove_recursive(replacement_node, swapped_locations, tracked_labels);

    } else {
        // --- Case 2: The node is a LEAF. ---
        // We can simply detach it from its parent.

        // Get a strong reference to the parent, if it exists.
        let parent_opt = node_to_remove.borrow().parent.upgrade();

        if let Some(parent_rc) = parent_opt {
            // The node has a parent, so find our node and set the parent's link to None.
            let mut parent_mut = parent_rc.borrow_mut();
            let direction = node_to_remove.borrow().direction_of_parent.clone();
            match direction {
                Direction::Left => parent_mut.left = None,
                Direction::Right => parent_mut.right = None,
            }
        } else {
            // No parent exists; this leaf node must be the root.
            // Deleting it makes the entire tree empty.
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

impl<TX: FloatNumber + RealNumber, X: Array2<TX>> AgglomerativeClustering<TX, X> {

/// Performs agglomerative clustering using a reciprocal nearest neighbor strategy.
pub fn fit(data: &X, n_clusters: usize) -> Result<Vec<usize>, String> {
    let (num_samples, num_features) = data.shape();
    if num_samples < n_clusters {
        return Err("Number of samples must be greater than or equal to n_clusters.".to_string());
    }
    // Convert data to a more convenient format.
    let data: Vec<Vec<f64>> = (0..num_samples)
        .map(|i| data.get_row(i).iterator(0).map(|x| x.to_f64().unwrap()).collect())
        .collect();

    // --- 1. Initialization ---
    let mut kdtree = KdTree::new(num_features);
    let mut clusters = HashMap::with_capacity(num_samples);

    // Each point starts as its own cluster. We store the node pointers,
    // though we only need them to begin the loop.
    kdtree.create(data.clone(), (0..num_samples).collect());
    for (i, row) in data.into_iter().enumerate() {
        clusters.insert(i, Cluster::new(i, row.clone()));
    }

    // --- 2. Main Clustering Loop ---
    // The state of our search is defined by a pair of clusters (A, B), where
    // B is the nearest neighbor of A. We check if A is also the nearest neighbor of B.
    //
    // A: The current cluster being evaluated.
    // B: A's nearest neighbor.
    // C: B's nearest neighbor.
    
    if num_samples <= n_clusters {
        // No merging needed, just assign unique labels.
        return Ok((0..num_samples).collect());
    }
    
    // --- Initialize the search with a starting pair (A, B) ---
    let mut search_index = 0;
    let (_, mut a_index, mut a_node) = kdtree.nearest(&clusters[&search_index].average, search_index);
    
    // Find the true nearest neighbor to A (the 1st result is A itself).
    let (mut distance_a_b, mut b_index, mut b_node) = kdtree.nearest(&clusters[&a_index].average, a_index);
;

    while clusters.len() > n_clusters {
        // Find C, the nearest neighbor of B.
        let (distance_b_c, c_index, c_node) = kdtree.nearest(&clusters[&b_index].average, b_index);
        // --- Reciprocal Match Check ---
        if distance_a_b <= distance_b_c {
            // MERGE: A and B are mutual nearest neighbors, a strong pair to merge.
            
            // Step 1: Remove A, but track if B's data gets swapped into A's place.
            // This is the most complex step. When deleting `a_node`, the k-d tree's
            // algorithm might move the data from `b_node` into `a_node`'s memory
            // location to fill the gap. If this happens, our `b_node` pointer is
            // still valid, but we must update it to point to its new location
            // before we try to remove it in the next step.
            let mut swapped_info = kdtree.remove(a_node.clone(), &HashSet::from([b_index]));
            if let Some(new_b_node_location) = swapped_info.remove(&b_index) {
                b_node = new_b_node_location;
            }

            // Step 2: Now that `b_node` is guaranteed to be correct, safely remove it.
            kdtree.remove(b_node.clone(), &HashSet::new());

            // Step 3: Merge the cluster data. Let the smaller ID absorb the larger one.
            let b_cluster = clusters.remove(&b_index).unwrap();
            let a_cluster_mut = clusters.get_mut(&a_index).unwrap();
            a_cluster_mut.add_cluster(b_cluster);

            // Step 4: Add the newly merged cluster back to the tree and get its new node pointer.
            let new_a_node = kdtree.add(a_cluster_mut.average.clone(), a_index);

            // Step 5: Update state for the next iteration.
            // The merged cluster is our new A. Find its nearest neighbor to get the new B.
            a_node = new_a_node;
            let (new_distance_ab, new_b_index, new_b_node) = kdtree.nearest(&clusters[&a_index].average, a_index);
            distance_a_b = new_distance_ab;
            b_index = new_b_index;
            b_node = new_b_node;

        } else {
            // WALK: Not a reciprocal match. Walk along the neighbor chain.
            // The old B becomes the new A, and its neighbor C becomes the new B.
            a_index = b_index;
            a_node = b_node;
            b_index = c_index;
            b_node = c_node;
            distance_a_b = distance_b_c;
        }
    }

    // --- 3. Final Label Assignment ---
    let mut labels = vec![0; num_samples];
    let mut cluster_indices: Vec<&usize> = clusters.keys().collect();
    cluster_indices.sort();
    for (final_label, cluster_index) in cluster_indices.into_iter().enumerate() {
        let cluster = clusters.get(cluster_index).unwrap();
        for original_point_index in &cluster.values {
            labels[*original_point_index] = final_label;
        }
    }

    Ok(labels)
}
}