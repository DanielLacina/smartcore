//! # Hierarchical Clustering
//!
//! Hierarchical clustering is a method of cluster analysis that builds a hierarchy of clusters, either from the bottom up or the top down. Unlike partitioning algorithms such as K-Means, it does not require the number of clusters to be specified beforehand. Instead, it joduces a tree-like structure called a dendrogram that illustrates the nested grouping of data points. A desired number of clusters can then be obtained by "cutting" the dendrogram at a specific level.
//!
//! This implementation uses the agglomerative (bottom-up) approach, which is the most common strategy for hierarchical clustering.
//!
//! The agglomerative algorithm works as follows:
//!
//! 1.  Initialization: Each data point starts in its own individual cluster.
//! 2.  Iterative Merging: In each step, the two closest clusters are identified and merged into a single new cluster.
//! 3.  Termination: This process is repeated until all data points are contained within a single, all-encompassing cluster, thus completing the hierarchy.
//!
//! A critical choice in this process is the linkage criterion, which defines how the distance between two clusters is measured. This choice significantly influences the shape of the clusters and the structure of the dendrogram. This implementation uses Ward's Linkage, which minimizes the increase in the total within-cluster variance when merging clusters. It is particularly effective at identifying compact, spherical clusters.
//!
//! Example:
//!
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::cluster::hierarchical::{AgglomerativeClustering, AgglomerativeClusteringParameters, Linkage};
//! let x = DenseMatrix::from_2d_array(&[
//!           &[5.1, 3.5, 1.4, 0.2],
//!           &[4.9, 3.0, 1.4, 0.2],
//!           &[4.7, 3.2, 1.3, 0.2],
//!           &[4.6, 3.1, 1.5, 0.2],
//!           &[5.0, 3.6, 1.4, 0.2],
//!           &[5.4, 3.9, 1.7, 0.4],
//!           &[4.6, 3.4, 1.4, 0.3],
//!           &[5.0, 3.4, 1.5, 0.2],
//!           &[4.4, 2.9, 1.4, 0.2],
//!           &[4.9, 3.1, 1.5, 0.1],
//!           &[7.0, 3.2, 4.7, 1.4],
//!           &[6.4, 3.2, 4.5, 1.5],
//!           &[6.9, 3.1, 4.9, 1.5],
//!           &[5.5, 2.3, 4.0, 1.3],
//!           &[6.5, 2.8, 4.6, 1.5],
//!           &[5.7, 2.8, 4.5, 1.3],
//!           &[6.3, 3.3, 4.7, 1.6],
//!           &[4.9, 2.4, 3.3, 1.0],
//!           &[6.6, 2.9, 4.6, 1.3],
//!           &[5.2, 2.7, 3.9, 1.4],
//!           &[6.3, 2.5, 5.0, 1.9],
//!           &[6.5, 3.0, 5.2, 2.0],
//!           &[6.2, 3.4, 5.4, 2.3],
//!           &[5.9, 3.0, 5.1, 1.8],
//!      ]).unwrap();
//! let params = AgglomerativeClusteringParameters {
//!     n_clusters: 3,
//!     linkage: Linkage::Ward,
//! };
//! let clustering_result = AgglomerativeClustering::<f64, f64, DenseMatrix<f64>, Vec<f64>>::fit(&x, params).unwrap();
//! let y_hat = clustering_result.labels;
//! ## References:
//!
//! * "An Introduction to Statistical Learning", James G., Witten D., Hastie T., Tibshirani R., Chapter 10
//! * "Hierarchical Grouping to Optimize an Objective Function", Ward, J. H., Jr., 1963
//! * "Finding Groups in Data: An Introduction to Cluster Analysis", Kaufman, L., Rousseeuw, P.J., 1990
use crate::api::UnsupervisedEstimator;
use crate::{
    error::Failed,
    linalg::basic::arrays::{Array1, Array2},
    numbers::basenum::Number,
};
use std::collections::HashMap;
use std::{f64, iter::zip, marker::PhantomData};

/// Defines the linkage criterion to use for Agglomerative Clustering.
///
/// The linkage criterion determines which distance to use between sets of observations.
/// The algorithm will merge the pairs of clusters that minimize this criterion.
pub enum Linkage {
    /// Ward's minimum variance method.
    ///
    /// Ward's method minimizes the sum of squared differences within all clusters.
    /// It is a variance-minimizing approach and in this sense is similar to the k-means
    /// objective function but tackled with an agglomerative hierarchical approach.
    Ward,
}

/// Parameters for the Agglomerative Clustering algorithm.
///
/// This struct is used to configure the clustering process. It can be instantiated
/// and then modified using a builder pattern.
pub struct AgglomerativeClusteringParameters {
    /// The number of clusters to find.
    pub n_clusters: usize,
    /// The linkage criterion to use.
    pub linkage: Linkage,
}

impl AgglomerativeClusteringParameters {
    /// Sets the number of clusters.
    ///
    /// # Arguments
    ///
    /// * `n_clusters` - The desired number of clusters.
    pub fn with_n_clusters(mut self, n_clusters: usize) -> Self {
        self.n_clusters = n_clusters;
        self
    }

    /// Sets the linkage criterion.
    ///
    /// # Arguments
    ///
    /// * `linkage` - The linkage method to use for clustering.
    pub fn with_linkage(mut self, linkage: Linkage) -> Self {
        self.linkage = linkage;
        self
    }
}

/// Represents the result of an Agglomerative Clustering operation.
///
/// This struct holds the cluster labels assigned to each sample in the input data.
pub struct AgglomerativeClustering<TX: Number, TY: Number, X: Array2<TX>, Y: Array1<TY>> {
    /// A vector where `labels[i]` is the cluster identifier for the i-th sample.
    pub labels: Vec<usize>,
    /// Phantom data to hold the generic type `TX`.
    _phantom_tx: PhantomData<TX>,
    /// Phantom data to hold the generic type `TY`.
    _phantom_ty: PhantomData<TY>,
    /// Phantom data to hold the generic type `X`.
    _phantom_x: PhantomData<X>,
    /// Phantom data to hold the generic type `Y`.
    _phantom_y: PhantomData<Y>,
}

impl<TX: Number, TY: Number, X: Array2<TX>, Y: Array1<TY>> AgglomerativeClustering<TX, TY, X, Y> {
    /// Computes the variance of a potential cluster.
    ///
    /// This function calculates the sum of squared distances from each point in the
    /// combined cluster to the cluster's mean. This is a key component of Ward's linkage.
    ///
    /// # Arguments
    ///
    /// * `data` - The input data matrix.
    /// * `cluster1_indices` - Indices of the data points in the first cluster.
    /// * `cluster2_indices` - Indices of the data points in the second cluster (can be empty).
    ///
    /// # Returns
    ///
    /// The variance of the combined cluster as an `f64`.
    fn compute_cluster_variance(
        data: &X,
        cluster1_indices: &Vec<usize>,
        cluster2_indices: &Vec<usize>,
    ) -> f64 {
        let (_, num_features) = data.shape();
        let mut sum_row = vec![0 as f64; num_features];

        // Sum up all feature vectors for the points in the given clusters
        for cluster in [cluster1_indices, cluster2_indices] {
            for index in cluster {
                sum_row = zip(sum_row, data.get_row(*index).iterator(0))
                    .map(|(v, x)| v + x.to_f64().unwrap())
                    .collect();
            }
        }

        let clusters_len = cluster1_indices.len() + cluster2_indices.len();
        // Calculate the mean of the combined cluster
        let mean_row: Vec<f64> = sum_row.iter().map(|v| *v / clusters_len as f64).collect();

        let mut variance = 0.0;
        // Calculate the sum of squared distances from each point to the mean
        for cluster in [cluster1_indices, cluster2_indices] {
            for index in cluster {
                let squared_distance: f64 = zip(data.get_row(*index).iterator(0), mean_row.iter())
                    .map(|(x, v)| (x.to_f64().unwrap() - *v).powf(2.0))
                    .sum::<f64>();
                variance += squared_distance;
            }
        }
        variance
    }

    /// Computes the distance between two clusters based on the specified linkage.
    ///
    /// # Arguments
    ///
    /// * `data` - The input data matrix.
    /// * `linkage` - The linkage criterion to use.
    /// * `cache` - A mutable HashMap to store and retrieve pre-computed cluster variances for performance.
    /// * `cluster1_indices` - Indices of the data points in the first cluster.
    /// * `cluster2_indices` - Indices of the data points in the second cluster.
    ///
    /// # Returns
    ///
    /// The distance between the two clusters as an `f64`.
    fn compute_distance(
        data: &X,
        linkage: &Linkage,
        cache: &mut HashMap<Vec<usize>, f64>,
        cluster1_indices: &Vec<usize>,
        cluster2_indices: &Vec<usize>,
    ) -> f64 {
        match linkage {
            Linkage::Ward => {
                // For Ward's method, the distance is the increase in variance that would result
                // from merging the two clusters.
                // distance = variance(cluster1 U cluster2) - variance(cluster1) - variance(cluster2)

                // Get variance of the first cluster, from cache or by computing it
                let cluster1_variance = if let Some(variance) = cache.get(cluster1_indices) {
                    *variance
                } else {
                    let cluster1_variance =
                        Self::compute_cluster_variance(data, cluster1_indices, &vec![]);
                    cache.insert(cluster1_indices.clone(), cluster1_variance);
                    cluster1_variance
                };

                // Get variance of the second cluster, from cache or by computing it
                let cluster2_variance = if let Some(variance) = cache.get(cluster2_indices) {
                    *variance
                } else {
                    let cluster2_variance =
                        Self::compute_cluster_variance(data, cluster2_indices, &vec![]);
                    cache.insert(cluster2_indices.clone(), cluster2_variance);
                    cluster2_variance
                };

                // Compute variance of the merged cluster
                let both_cluster_variance =
                    Self::compute_cluster_variance(data, cluster1_indices, cluster2_indices);

                // The increase in variance is the distance
                both_cluster_variance - cluster1_variance - cluster2_variance
            }
        }
    }

    /// Fit the agglomerative clustering model to the data.
    ///
    /// This method performs hierarchical clustering using a bottom-up approach. Each observation
    /// starts in its own cluster, and clusters are successively merged together. The process
    /// continues until the desired number of clusters is reached.
    ///
    /// # Arguments
    ///
    /// * `data` - A 2D array-like structure of shape (n_samples, n_features).
    /// * `parameters` - The parameters for the clustering algorithm, including `n_clusters` and `linkage`.
    ///
    /// # Returns
    ///
    /// A `Result` which is `Ok` containing an `AgglomerativeClustering` instance with the
    /// final cluster labels, or an `Err` with a `Failed` error type if something goes wrong.
    pub fn fit(
        data: &X,
        parameters: AgglomerativeClusteringParameters,
    ) -> Result<AgglomerativeClustering<TX, TY, X, Y>, Failed> {
        let mut cache = HashMap::new();
        let mut matrix = Vec::new();
        let (num_rows, _) = data.shape();

        // Initially, each data point is its own cluster.
        // `indices_mapping` maps a cluster ID to the list of original data point indices it contains.
        let mut indices_mapping = HashMap::new();
        for i in 0..num_rows {
            indices_mapping.insert(i, vec![i]);
        }

        // Pre-compute the initial distance matrix for all pairs of points.
        // This is an upper triangular matrix to save space.
        for i in 0..num_rows {
            let mut row = Vec::new();
            for j in i + 1..num_rows {
                let distance = Self::compute_distance(
                    data,
                    &parameters.linkage,
                    &mut cache,
                    indices_mapping.get(&i).unwrap(),
                    indices_mapping.get(&j).unwrap(),
                );
                row.push(distance);
            }
            matrix.push(row);
        }

        // Iteratively merge clusters until `n_clusters` is reached.
        while indices_mapping.len() > parameters.n_clusters {
            let mut min_distance = f64::INFINITY;
            let mut pairs = (0, 0);

            // Find the two closest clusters.
            for (i, row) in matrix.iter().enumerate() {
                if !indices_mapping.contains_key(&i) {
                    continue; // Skip clusters that have been merged.
                }
                for (j, distance) in row.iter().enumerate() {
                    let j_offset = i + 1 + j; // Get the real index for the second cluster.
                    if !indices_mapping.contains_key(&j_offset) {
                        continue; // Skip clusters that have been merged.
                    }
                    if *distance < min_distance {
                        min_distance = *distance;
                        pairs = (i, j_offset);
                    }
                }
            }

            let (i, j_offset) = pairs;

            // Merge the two closest clusters (`i` and `j_offset`).
            let cluster1_indices = indices_mapping.remove(&i).unwrap();
            let cluster2_indices = indices_mapping.remove(&j_offset).unwrap();
            cache.remove(&cluster1_indices); // Clear old cache entries.
            cache.remove(&cluster2_indices);

            let mut combined_cluster_indices = cluster1_indices;
            combined_cluster_indices.extend(cluster2_indices);

            // Update the distance matrix. The new merged cluster will be stored at index `i`.
            // Update distances from the new cluster `i` to all other clusters `j` where `j > i`.
            matrix[i] = (0..matrix[i].len())
                .map(|j| {
                    let j_offset = i + 1 + j;
                    if let Some(other_cluster_indices) = indices_mapping.get(&j_offset) {
                        Self::compute_distance(
                            data,
                            &parameters.linkage,
                            &mut cache,
                            &combined_cluster_indices,
                            other_cluster_indices,
                        )
                    } else {
                        0.0 // This entry is now invalid as the other cluster was merged.
                    }
                })
                .collect();

            #[allow(clippy::needless_range_loop)]
            // Update distances from all other clusters `g` to the new cluster `i` where `g < i`.
            for g in 0..i {
                let offset = i - g - 1;
                if let Some(other_cluster_indices) = indices_mapping.get(&g) {
                    matrix[g][offset] = Self::compute_distance(
                        data,
                        &parameters.linkage,
                        &mut cache,
                        &combined_cluster_indices, // Order does not matter for Ward's method.
                        other_cluster_indices,
                    )
                }
            }
            // Add the new merged cluster to the mapping.
            indices_mapping.insert(i, combined_cluster_indices);
        }

        // Assign final labels based on the remaining clusters.
        let mut labels = vec![0; num_rows];
        let mut sorted_keys: Vec<&usize> = indices_mapping.keys().collect();
        sorted_keys.sort(); // Sort for consistent label assignment.
        for (i, cluster) in sorted_keys.iter().enumerate() {
            for index in indices_mapping.get(cluster).unwrap() {
                labels[*index] = i;
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

impl<TX: Number, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    UnsupervisedEstimator<X, AgglomerativeClusteringParameters>
    for AgglomerativeClustering<TX, TY, X, Y>
{
    fn fit(x: &X, parameters: AgglomerativeClusteringParameters) -> Result<Self, Failed> {
        AgglomerativeClustering::fit(x, parameters)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::matrix::DenseMatrix;
    use std::collections::HashSet;

    fn assert_approx_eq(a: f64, b: f64) {
        assert!(
            (a - b).abs() < 1e-6,
            "assertion failed: `(left !== right)` \n left: `{:?}`\n right: `{:?}`",
            a,
            b
        );
    }

    #[test]
    fn test_compute_cluster_variance() {
        let data = DenseMatrix::from_2d_array(&[&[1.0, 1.0], &[3.0, 3.0], &[5.0, 5.0]]).unwrap();

        // Variance of a single point is 0
        let variance1 =
            AgglomerativeClustering::<f64, f64, DenseMatrix<f64>, Vec<f64>>::compute_cluster_variance(
                &data,
                &vec![0],
                &vec![],
            );
        assert_approx_eq(variance1, 0.0);

        // Variance of two points: [1,1] and [3,3]
        // Mean is [2,2]
        // Variance = ((1-2)^2 + (1-2)^2) + ((3-2)^2 + (3-2)^2) = (1+1) + (1+1) = 4.0
        let variance2 =
            AgglomerativeClustering::<f64, f64, DenseMatrix<f64>, Vec<f64>>::compute_cluster_variance(
                &data,
                &vec![0],
                &vec![1],
            );
        assert_approx_eq(variance2, 4.0);

        // Variance of three points: [1,1], [3,3], [5,5]
        // Mean is [3,3]
        // Variance = ((1-3)^2+(1-3)^2) + ((3-3)^2+(3-3)^2) + ((5-3)^2+(5-3)^2)
        //          = (4+4) + (0+0) + (4+4) = 16.0
        let variance3 =
            AgglomerativeClustering::<f64, f64, DenseMatrix<f64>, Vec<f64>>::compute_cluster_variance(
                &data,
                &vec![0, 1, 2],
                &vec![],
            );
        assert_approx_eq(variance3, 16.0);
    }

    #[test]
    fn test_compute_distance_ward() {
        let data = DenseMatrix::from_2d_array(&[&[1.0, 1.0], &[3.0, 3.0]]).unwrap();
        let mut cache = HashMap::new();

        let cluster1_indices = vec![0];
        let cluster2_indices = vec![1];

        // var(c1) = 0, var(c2) = 0
        // var(c1 U c2) = 4.0 (from test above)
        // distance = 4.0 - 0 - 0 = 4.0
        let distance =
            AgglomerativeClustering::<f64, f64, DenseMatrix<f64>, Vec<f64>>::compute_distance(
                &data,
                &Linkage::Ward,
                &mut cache,
                &cluster1_indices,
                &cluster2_indices,
            );

        assert_approx_eq(distance, 4.0);
        // check that cache was populated
        assert!(cache.contains_key(&cluster1_indices));
        assert!(cache.contains_key(&cluster2_indices));
    }

    #[test]
    fn test_fit_simple_clusters() {
        let data = DenseMatrix::from_2d_array(&[
            &[1.0, 2.0],  // cluster 0
            &[1.5, 1.8],  // cluster 0
            &[1.0, 0.6],  // cluster 0
            &[8.0, 8.0],  // cluster 1
            &[9.0, 11.0], // cluster 1
            &[8.5, 9.5],  // cluster 1
        ])
        .unwrap();

        let params = AgglomerativeClusteringParameters {
            n_clusters: 2,
            linkage: Linkage::Ward,
        };

        let result =
            AgglomerativeClustering::<f64, f64, DenseMatrix<f64>, Vec<f64>>::fit(&data, params)
                .unwrap();
        let labels = result.labels;

        assert_eq!(labels.len(), 6);

        let label_set_1 = labels[0];
        let label_set_2 = labels[3];

        // Assert the two sets have different labels
        assert_ne!(label_set_1, label_set_2);

        // Assert that the first three points belong to the same cluster
        assert_eq!(labels[0], label_set_1);
        assert_eq!(labels[1], label_set_1);
        assert_eq!(labels[2], label_set_1);

        // Assert that the last three points belong to the same cluster
        assert_eq!(labels[3], label_set_2);
        assert_eq!(labels[4], label_set_2);
        assert_eq!(labels[5], label_set_2);
    }

    #[test]
    fn test_n_clusters_parameter() {
        let data =
            DenseMatrix::from_2d_array(&[&[0.0], &[1.0], &[10.0], &[11.0], &[20.0], &[21.0]])
                .unwrap();

        // Test with n_clusters = 3
        let params_3 = AgglomerativeClusteringParameters {
            n_clusters: 3,
            linkage: Linkage::Ward,
        };
        let result_3 =
            AgglomerativeClustering::<f64, f64, DenseMatrix<f64>, Vec<f64>>::fit(&data, params_3)
                .unwrap();
        let unique_labels_3: HashSet<usize> = result_3.labels.into_iter().collect();
        assert_eq!(unique_labels_3.len(), 3);

        // Test with n_clusters = 1
        let params_1 = AgglomerativeClusteringParameters {
            n_clusters: 1,
            linkage: Linkage::Ward,
        };
        let result_1 =
            AgglomerativeClustering::<f64, f64, DenseMatrix<f64>, Vec<f64>>::fit(&data, params_1)
                .unwrap();
        let unique_labels_1: HashSet<usize> = result_1.labels.into_iter().collect();
        assert_eq!(unique_labels_1.len(), 1);
    }

    #[test]
    fn test_fit_heavy_load_deterministic() {
        let n_clusters = 5;

        // Define cluster properties: (center_x, center_y, num_points)
        let cluster_definitions = vec![
            (0.0, 0.0, 10),
            (100.0, 0.0, 20),
            (0.0, 100.0, 15),
            (100.0, 100.0, 25),
            (50.0, -50.0, 5),
        ];

        // The expected sizes of the final clusters.
        let mut expected_counts: Vec<usize> = cluster_definitions.iter().map(|c| c.2).collect();
        expected_counts.sort_unstable();

        let mut data_vec: Vec<Vec<f64>> = Vec::new();

        // Generate data points for each cluster deterministically.
        for (center_x, center_y, num_points) in cluster_definitions {
            for i in 0..num_points {
                // Add a small, predictable offset to each point based on its index.
                // This creates a small, non-random spread around the center.
                let offset = i as f64 * 0.1;
                let x = center_x + offset;
                let y = center_y + offset;
                data_vec.push(vec![x, y]);
            }
        }

        // Convert to DenseMatrix
        let data_refs: Vec<&[f64]> = data_vec.iter().map(|row| row.as_slice()).collect();
        let data = DenseMatrix::from_2d_array(&data_refs).unwrap();

        // Run clustering
        let params = AgglomerativeClusteringParameters {
            n_clusters,
            linkage: Linkage::Ward,
        };
        let result =
            AgglomerativeClustering::<f64, f64, DenseMatrix<f64>, Vec<f64>>::fit(&data, params)
                .unwrap();
        let labels = result.labels;

        // 1. Verify the number of distinct clusters found
        let unique_labels: HashSet<usize> = labels.iter().cloned().collect();
        assert_eq!(
            unique_labels.len(),
            n_clusters,
            "Expected {} distinct clusters, but found {}",
            n_clusters,
            unique_labels.len()
        );

        // 2. Verify the number of members in each cluster
        let mut label_counts: HashMap<usize, usize> = HashMap::new();
        for label in labels {
            *label_counts.entry(label).or_insert(0) += 1;
        }

        let mut actual_counts: Vec<usize> = label_counts.values().cloned().collect();
        actual_counts.sort_unstable();

        assert_eq!(
            actual_counts, expected_counts,
            "Cluster sizes do not match expected values"
        );
    }
}
