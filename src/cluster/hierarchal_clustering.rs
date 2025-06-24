use crate::{
    error::Failed,
    linalg::basic::arrays::{Array, Array1, Array2},
    metrics::distance::euclidian::Euclidian,
    numbers::basenum::Number,
};
use std::collections::HashMap;
use std::{f32, iter::zip, marker::PhantomData};

pub enum Linkage {
    Ward,
}

pub struct AgglomerativeClusteringParameters {
    pub n_clusters: usize,
    pub linkage: Linkage,
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

pub struct AgglomerativeClustering<TX: Number, TY: Number, X: Array2<TX>, Y: Array1<TY>> {
    pub labels: Vec<usize>,
    _phantom_tx: PhantomData<TX>,
    _phantom_ty: PhantomData<TY>,
    _phantom_x: PhantomData<X>,
    _phantom_y: PhantomData<Y>,
}

impl<TX: Number, TY: Number, X: Array2<TX>, Y: Array1<TY>> AgglomerativeClustering<TX, TY, X, Y> {
    fn compute_cluster_variance(
        data: &X,
        cluster1_indices: &Vec<usize>,
        cluster2_indices: &Vec<usize>,
    ) -> f32 {
        let (_, num_features) = data.shape();
        let mut sum_row = vec![0 as f32; num_features];
        for cluster in vec![cluster1_indices, cluster2_indices] {
            for index in cluster {
                sum_row = zip(sum_row, data.get_row(*index).iterator(0))
                    .map(|(v, x)| v + x.to_f32().unwrap())
                    .collect();
            }
        }
        let clusters_len = cluster1_indices.len() + cluster2_indices.len();
        let mean_row: Vec<f32> = sum_row.iter().map(|v| *v/clusters_len as f32).collect(); 
        let mut variance = 0.0;
        for cluster in vec![cluster1_indices, cluster2_indices] {
            for index in cluster {
                let squared_distance: f32 = zip(data.get_row(*index).iterator(0), mean_row.iter())
                    .map(|(x, v)| (x.to_f32().unwrap() - *v).powf(2.0))
                    .sum();
                variance += squared_distance;
            }
        }
        variance
    }

    fn compute_distance<'a>(
        data: &X,
        linkage: &Linkage,
        cache: &mut HashMap<&'a Vec<usize>, f32>,
        cluster1_indices: &'a Vec<usize>,
        cluster2_indices: &'a Vec<usize>,
    ) -> f32 {
        match linkage {
            Linkage::Ward => {
                let cluster1_variance = if let Some(variance) = cache.get(&cluster1_indices) {
                    *variance
                } else {
                    let cluster1_variance =
                        Self::compute_cluster_variance(&data, &cluster1_indices, &vec![]);
                    cache.insert(&cluster1_indices, cluster1_variance);
                    cluster1_variance
                };
                let cluster2_variance = if let Some(variance) = cache.get(&cluster2_indices) {
                    *variance
                } else {
                    let cluster2_variance =
                        Self::compute_cluster_variance(&data, &cluster2_indices, &vec![]);
                    cache.insert(&cluster2_indices, cluster2_variance);
                    cluster2_variance
                };
                let both_cluster_variance = cluster1_variance + cluster2_variance;
                let distance = both_cluster_variance - cluster1_variance - cluster2_variance;
                distance
            }
        }
    }
    pub fn fit(
        data: &X,
        parameters: AgglomerativeClusteringParameters,
    ) -> Result<AgglomerativeClustering<TX, TY, X, Y>, Failed> {
        let mut cache = HashMap::new();
        let mut matrix = Vec::new();
        let (num_rows, _) = data.shape();
        let mut indices_mapping = HashMap::new();
        for i in 0..num_rows {
            indices_mapping.insert(i, vec![i]);
        }
        for i in 0..num_rows {
            let mut row = Vec::new();
            for j in i + 1..num_rows {
                let distance = Self::compute_distance(
                    data,
                    &parameters.linkage,
                    &mut cache,
                    indices_mapping.get(&i).unwrap(),
                    &indices_mapping.get(&j).unwrap(),
                );
                row.push(distance);
            }
            matrix.push(row);
        }
        while indices_mapping.len() > parameters.n_clusters {
            let mut min_distance = f32::INFINITY;
            let mut pairs = (0, 0);
            for (i, row) in matrix.iter().enumerate() {
                if !indices_mapping.contains_key(&i) {
                    continue;
                }
                for (j, distance) in row.iter().enumerate() {
                    let j_offset = i + 1 + j;
                    if !indices_mapping.contains_key(&j_offset) {
                        continue;
                    }
                    if *distance < min_distance {
                        min_distance = *distance;
                        pairs = (i, j_offset);
                    }
                }
            }
            let (i, j_offset) = pairs;
            let cluster1_indices = indices_mapping.remove(&i).unwrap();
            let cluster2_indices = indices_mapping.remove(&j_offset).unwrap();
            cache.remove(&cluster1_indices);
            cache.remove(&cluster2_indices);
            let mut combined_cluster_indices = cluster1_indices;
            combined_cluster_indices.extend(cluster2_indices);
            indices_mapping.insert(i, combined_cluster_indices);
            matrix[i] = (0..matrix[i].len())
                .map(|j| {
                    if let Some(other_cluster_indices) = indices_mapping.get(i + 1 + j) {
                        Self::compute_distance(
                            &data,
                            &parameters.linkage,
                            &mut cache,
                            &combined_cluster_indices,
                            &other_cluster_indices,
                        )
                    } else {
                        0.0
                    }
                })
                .collect();
            for g in 0..i {
                let offset = i - g - 1;
                if let Some(other_cluster_indices) = indices_mapping.get(&offset) {
                    matrix[g][offset] = Self::compute_distance(
                        &data,
                        &parameters.linkage,
                        &mut cache,
                        &combined_cluster_indices,
                        &other_cluster_indices,
                    )
                }
            }
        }
        let mut labels = vec![0; num_rows];
        for (i, cluster) in indices_mapping.keys().enumerate() {
            for index in indices_mapping[cluster] {
                labels[index] = i
            }
        }
        Ok(Self {
            labels,
            _phantom_tx: PhantomData,
            _phantom_ty: PhantomData,
            _phantom_x: PhantomData,
            _phantom_y: PhantomData,
        })
    }
}
