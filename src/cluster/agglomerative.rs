use std::{collections::BinaryHeap, marker::PhantomData};
use std::cmp::Reverse;
use crate::{error::{Failed, FailedError}, linalg::basic::arrays::{Array1, Array2, ArrayView1}, metrics::distance::euclidian::Euclidian, numbers::{basenum::Number, floatnum::FloatNumber, realnum::RealNumber}};


pub enum Linkage {
    Single,
    Complete,
    Average,
    Ward,
} 

pub struct AgglomerativeClusteringParameters {
    pub n_clusters: usize,
    pub linkage: Linkage,
} 

impl Default for AgglomerativeClusteringParameters {
    fn default() -> Self {
        AgglomerativeClusteringParameters { n_clusters: 2, linkage: Linkage::Ward }
    }

}

impl AgglomerativeClusteringParameters {
    pub fn with_n_clusters(mut self, n_clusters: usize) -> Self {
        self.n_clusters = n_clusters;
        self
    }

    pub fn with_linkage(mut self, linkage: Linkage) -> Self {
        self.linkage = linkage;
        self
    }
}


pub struct AgglomerativeClustering<TX: RealNumber + FloatNumber, TY: Number, X: Array2<TX>, Y: Array1<TY>> {
    pub labels: Vec<usize>, 
    _phantom_tx: PhantomData<TX>,
    _phantom_ty: PhantomData<TY>,
    _phantom_x: PhantomData<X>,
    _phantom_y: PhantomData<Y>,
}

impl<TX: RealNumber + FloatNumber, TY: Number, X: Array2<TX>, Y: Array1<TY>> AgglomerativeClustering<TX, TY, X, Y> {
    fn compute_updated_distance(
    linkage: &Linkage,
    dist_ik: f64,
    dist_jk: f64,
    dist_ij: f64,
    size_i: usize,
    size_j: usize,
    size_k: usize,
) -> f64 {
    // Determine the Lance-Williams formula coefficients based on the linkage method.
    let (alpha_i, alpha_j, beta, gamma) = match linkage {
        Linkage::Single => (0.5, 0.5, 0.0, -0.5),
        Linkage::Complete => (0.5, 0.5, 0.0, 0.5),
        Linkage::Average => {
            let size_i = size_i as f64;
            let size_j = size_j as f64;
            let total_size = size_i + size_j;
            (size_i / total_size, size_j / total_size, 0.0, 0.0)
        }
        Linkage::Ward => {
            let size_i = size_i as f64;
            let size_j = size_j as f64;
            let size_k = size_k as f64;
            let n_total = size_i + size_j + size_k;
            (
                (size_i + size_k) / n_total,
                (size_j + size_k) / n_total,
                -size_k / n_total,
                0.0,
            )
        }
    };

    // The general Lance-Williams formula.
    alpha_i * dist_ik + alpha_j * dist_jk + beta * dist_ij + gamma * (dist_ik - dist_jk).abs()
}
    pub fn fit(data: &X, parameters: AgglomerativeClusteringParameters) -> Result<Self, Failed> {
        let (num_rows, _) = data.shape();
        let n_clusters = parameters.n_clusters;
        let linkage = &parameters.linkage;
        if n_clusters > num_rows {
            return Err(Failed::because(FailedError::ParametersError, "Number of clusters cannot be greater than number of data points."));
        } 
        let mut distances = HashMap::new();
        let mut heap = BinaryHeap::new();
        // maps to cluster ids
        let mut cluster_assignments = HashMap::new();
        // maps to oroiginal indices
        let mut cluster_indices = HashMap::new();

        for i in 0..num_rows {
            cluster_assignments.insert(i, vec![i]);
            cluster_indices.insert(i, vec![i]);
        }

        for i in 0..num_rows {
            for j in 0..num_rows {
                let distance: f64 = data.get_row(i).iterator(0).zip(data.get_row(j).iterator(0)).map(|(ci, cj)| (ci.to_f64().unwrap() - ci.to_f64().unwrap()).powi(2)).sum();
                distances.insert(vec![i, j], distance);
                heap.push((Reverse(((distance * 10e9) as usize, i, j))));
            }
        } 

        let mut cluster_id = num_rows;

        while cluster_assignments.len() > n_clusters {
            let Reverse((distance_ij, i, j)) = heap.pop().unwrap();
            let distance_ij = distance_ij as f64 / 10e9;
            let (size_i, size_j) = (cluster_indices[i].len(), cluster_indices[j].len());

            for k in cluster_assignments.keys() {
                if k == i || k == j {
                    continue;
                } 
                let size_k = cluster_indices[k].len();
                let ik_key = vec![i, k].sort_by(|a, b| a.cmp(b)); 
                let jk_key = vec![j, k].sort_by(|a, b| a.cmp(b));
                let distance_ik = distances.get(&ik_key).unwrap(); 
                let distance_jk = distances.get(&jk_key).unwrap();
                let new_distance =  Self::compute_updated_distance(linkage, distance_ik, distance_jk, distance_ij, size_i, size_j, size_k)
                let new_key = vec![cluster_id, k].sort_by(|a, b| a.cmp(b));
                distances.insert(new_key, new_distance);
                heap.push(Reverse(((new_distance * 10e9) as usize, cluster_id, k)));
            }
            let mut cluster_indices_i = cluster_indices.remove(&i).unwrap();
            let cluster_indices_j = cluster_indices.remove(&j).unwrap();
            cluster_indices_i.extend(cluster_indices_j);
            let new_cluster_indices = cluster_indices_i;
            cluster_indices.insert(cluster_id, new_cluster_indices);
            cluster_assignments.remove(&i);
            cluster_assignments.remove(&j);
            cluster_assignments.insert(cluster_id, vec![i, j]);
            cluster_id += 1;
        }

        let mut labels: Vec<usize> = (0..num_rows).map(|_| 0).collect(); 
        let cluster_keys: Vec<&usize> = cluster_indices.keys().collect();
        for (i, key) in cluster_keys.iter().enumerate() {
            for &index in cluster_indices.get(key).unwrap() {
                labels[index] = i;
            }
        }
         
        Ok(AgglomerativeClustering {
            labels,
            _phantom_tx: PhantomData,
            _phantom_ty: PhantomData,
            _phantom_x: PhantomData,
            _phantom_y: PhantomData,
        })

    }
}