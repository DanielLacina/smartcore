use std::collections::{HashMap};
use crate::linalg::basic::arrays::{Array1, Array2};
use crate::metrics::distance::euclidian::Euclidian;
use crate::metrics::distance::PairwiseDistance;
use crate::numbers::floatnum::FloatNumber;
use crate::error::{Failed, FailedError};
use crate::numbers::realnum::RealNumber;
use std::marker::PhantomData;
use std::fmt;

pub struct LinkageNode {
    pub is_connector: bool,
    pub index: usize,     
    pub left: Option<Box<LinkageNode>>,
    pub right: Option<Box<LinkageNode>>,
    pub distance: f64
}

impl fmt::Debug for LinkageNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut builder = f.debug_struct(if self.is_connector { "Connector" } else { "Leaf" });

        builder
            .field("index", &self.index)
            .field("distance", &format!("{:.2}", self.distance)); // Format distance here

        // Conditionally add left and right children
        if self.left.is_some() {
            builder.field("left", &self.left); // Recurses on Debug impl of Box<LinkageNode>
        }
        if self.right.is_some() {
            builder.field("right", &self.right); // Recurses on Debug impl of Box<LinkageNode>
        }

        builder.finish()
    }
}


impl LinkageNode {
    pub fn new(index: usize, distance: f64) -> Self {
        Self {
            index,
            is_connector: false,
            left: None,
            right: None,
            distance
        } 
    }
    pub fn connector_node(left: LinkageNode, right: LinkageNode) -> Self {
        Self {
            index: 0,
            is_connector: true,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            distance: 0.0
        }
    }
}


pub struct AgglomerativeClustering<TX: FloatNumber, X: Array2<TX>> {
    pub dendrogram: LinkageNode,
    pub labels: Vec<usize>,
    /// Phantom data to hold the generic type `TX`.
    _phantom_tx: PhantomData<TX>,
    /// Phantom data to hold the generic type `X`.
    _phantom_x: PhantomData<X>,
}

impl<TX: FloatNumber + RealNumber, X: Array2<TX>> AgglomerativeClustering<TX, X> {
    fn find_and_sort_distances_by_max_to_min(data: &X) -> Vec<PairwiseDistance<f64>> {
      let (num_samples, num_features) = data.shape();
        let mut distances = Vec::new(); 
        for index_row_i in 0..(num_samples) {
            for index_row_j in (index_row_i + 1)..num_samples {
                let d = Euclidian::squared_distance(
                    &Vec::from_iterator(
                        data.get_row(index_row_i).iterator(0).copied(),
                        num_features,
                    ),
                    &Vec::from_iterator(
                        data.get_row(index_row_j).iterator(0).copied(),
                        num_features
                    ),
                );
                distances.push(PairwiseDistance {
                    node: index_row_i,
                    neighbour: Some(index_row_j),
                    distance: Some(d)
                });
             }
        }
        distances.sort_by(|a, b| b.partial_cmp(a).unwrap());
        distances

    }
    pub fn fit(data: &X) -> Result<Self, Failed> {
        let (num_samples, _) = data.shape();
        let mut distances = Self::find_and_sort_distances_by_max_to_min(data);         
        let mut index_to_stack_id  = HashMap::with_capacity(num_samples);
        let mut stack = HashMap::new(); 
        let mut stack_id = 0;
        while let Some(pair) = distances.pop() {
            let index1 = pair.node;      
            let index2 = pair.neighbour.unwrap();
            let distance = pair.distance.unwrap();
            let index1_stack_id = index_to_stack_id.get(&index1).map(|stack_id_ref| *stack_id_ref);
            let index2_stack_id = index_to_stack_id.get(&index2).map(|stack_id_ref| *stack_id_ref);
            if index1_stack_id.is_none() && index2_stack_id.is_none() {
                let connector_node = LinkageNode::connector_node(LinkageNode::new(index1, distance), LinkageNode::new(index2, distance));
                stack.insert(stack_id, (Some(connector_node), vec![index1, index2]));
                for index in [index1, index2] {
                    index_to_stack_id.insert(index, stack_id);
                }
                stack_id += 1;
            } else if index1_stack_id.is_some() && index2_stack_id.is_some() {
                let index1_stack_id = index1_stack_id.unwrap(); 
                let index2_stack_id = index2_stack_id.unwrap(); 
                if index1_stack_id == index2_stack_id {
                    continue;
                }
                let node1= {
                    let (node1, _) = stack.get_mut(&index1_stack_id).unwrap();
                    node1.take().unwrap()
                };
                let node2 = {
                    let (node2, _) = stack.get_mut(&index2_stack_id).unwrap();
                    node2.take().unwrap()
                }; 
                // merge cluster associated with index2 with cluster associated with index1
                let connector_node = LinkageNode::connector_node(node1, node2);  
                let (_, node_indices2) = stack.remove(&index2_stack_id).unwrap();
                let (node1, node_indices1) = stack.get_mut(&index1_stack_id).unwrap();
                *node1 = Some(connector_node);
                for index in node_indices2.iter() {
                    index_to_stack_id.insert(*index, index1_stack_id); 
                }  
                node_indices1.extend_from_slice(node_indices2.as_slice());
            } 
            else  {
                let [stack_id, index_not_in_stack] = if let Some(stack_id) = index1_stack_id {
                    [stack_id, index2]
                } else if let Some(stack_id) = index2_stack_id {
                    [stack_id, index1]
                } else {
                    return Err(Failed::because(
                        FailedError::InvalidStateError,
                     "one of the stack ids should be Some()"
                    ));
                };
                let (node, node_indices) = stack.get_mut(&stack_id).unwrap();
                let node_unwrapped = node.as_mut().unwrap();
                if node_unwrapped.is_connector {
                    node_unwrapped.is_connector = false;
                    node_unwrapped.index = index_not_in_stack;
                    node_unwrapped.distance = distance;
                } else {
                    let mut new_parent_node = LinkageNode::new(index_not_in_stack, distance); 
                    new_parent_node.left = Some(Box::new(node.take().unwrap()));
                    *node = Some(new_parent_node);
                }
                node_indices.push(index_not_in_stack);
                index_to_stack_id.insert(index_not_in_stack,stack_id);
            }  
        }
        let final_stack_id = index_to_stack_id.get(&0).unwrap();
        let dendrogram = stack.remove(final_stack_id).unwrap().0.unwrap();
        Ok(Self {
            labels: Vec::new(),
            dendrogram,
            _phantom_tx: PhantomData,
            _phantom_x: PhantomData
        })
    }
}