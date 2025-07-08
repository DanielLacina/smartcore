use std::{iter::zip, marker::PhantomData};
use rand::{seq::SliceRandom, Rng};

use crate::{
    api::{PredictorBorrow, SupervisedEstimatorBorrow},
    error::{Failed, FailedError},
    linalg::basic::arrays::{Array1, Array2},
    numbers::{basenum::Number, realnum::RealNumber},
    rand_custom::get_rng_impl,
    tree::decision_tree_regressor::{self, DecisionTreeRegressor, DecisionTreeRegressorParameters},
};

struct TreeBooster {
    left: Option<Box<TreeBooster>>,
    right: Option<Box<TreeBooster>>,
    value: f64,
    threshold: f64,
    split_feature_idx: usize,
    split_score: f64
}

impl TreeBooster {
    pub fn fit(
        data: &Vec<Vec<f64>>,
        g: &Vec<f64>,
        h: &Vec<f64>,
        idxs: &Vec<usize>,
        max_depth: u16,
        min_child_weight: f64,
        lambda: f64,
        gamma: f64,
    ) -> Self{
        let value = g.iter().sum::<f64>() / (h.iter().sum::<f64>() + lambda);
        let mut best_feature_idx = usize::MAX;
        let mut best_split_score = 0.0;
        let mut best_threshold = 0.0;
        let mut left = Option::None;
        let mut right = Option::None;
        if max_depth <= 0 {
            return Self {
                left,
                right,
                value,
                threshold: best_threshold,
                split_feature_idx: best_feature_idx,
                split_score: best_split_score,
            };     
        } 
        Self::insert_child_nodes(data, g, h, idxs, &mut best_feature_idx, &mut best_split_score, &mut best_threshold, &mut left, &mut right, max_depth, min_child_weight, lambda, gamma); 
        Self {
            left,
            right,
            value,
            threshold: best_threshold,
            split_feature_idx: best_feature_idx,
            split_score: best_split_score,
        }
    }

    fn insert_child_nodes (
        data: &Vec<Vec<f64>>,
        g: &Vec<f64>,
        h: &Vec<f64>,
        idxs: &Vec<usize>,
        best_feature_idx: &mut usize,
        best_split_score: &mut f64, 
        best_threshold: &mut f64, 
        left: &mut Option<Box<Self>>, 
        right: &mut Option<Box<Self>>,
        max_depth: u16,
        min_child_weight: f64,
        lambda: f64,
        gamma: f64) {
        for i in 0..data[0].len() {
            Self::find_best_split(data, g, h, idxs, i, best_feature_idx, best_split_score, best_threshold, min_child_weight, lambda, gamma);
        }      
        if *best_split_score == 0.0 {
            return;
        } 
        let mut left_idxs = Vec::new();
        let mut right_idxs = Vec::new();
        for idx in idxs.iter() {
            if data[*idx][*best_feature_idx] <= *best_threshold {
                left_idxs.push(*idx);
            } else {
                right_idxs.push(*idx);
            }
        }
        *left = Some(Box::new(TreeBooster::fit(
            data,
            g,
            h,
            &left_idxs,
            max_depth - 1,
            min_child_weight,
            lambda,
            gamma,
        )));
        *right = Some(Box::new(TreeBooster::fit(
            data,
            g,
            h,
            &right_idxs,
            max_depth - 1,
            min_child_weight,
            lambda,
            gamma,
        )));
    }
    
    fn find_best_split(
        data: &Vec<Vec<f64>>,
        g: &Vec<f64>,
        h: &Vec<f64>,
        idxs: &Vec<usize>,
        feature_idx: usize,
        best_feature_idx: &mut usize,
        best_split_score: &mut f64, 
        best_threshold: &mut f64, 
        min_child_weight: f64,
        lambda: f64,
        gamma: f64,
    ) {
        let mut idxs = idxs.clone();
        idxs.sort_by(|a, b| data[*a][feature_idx].partial_cmp(&data[*b][feature_idx]).unwrap());
        let sum_g = idxs.iter().map(|&i| g[i]).sum::<f64>();
        let sum_h = idxs.iter().map(|&i| h[i]).sum::<f64>();
        let (mut sum_g_right, mut sum_h_left) = (sum_g, sum_h);
        let (mut sum_g_left, mut sum_h_right) = (0.0, 0.0);
        for i in 0..idxs.len() - 1 {
           let idx = idxs[i]; 
           let (g_i, h_i, x_i, x_i_next) = (g[idx],h[idx],data[idx][feature_idx], data[idx][feature_idx]); 
           sum_g_left += g_i;
           sum_h_left += h_i;
           sum_g_right -= g_i;
           sum_h_right -= h_i;
           if sum_h_left < min_child_weight || x_i == x_i_next {
            continue;
           }
           if sum_h_right < min_child_weight {
            break;
           }
           let gain = 0.5 * (sum_g_left * sum_g_left / (sum_h_left + lambda) + sum_g_right * sum_g_right / (sum_h_right + lambda) - sum_g * sum_g / (sum_h + lambda))
                - gamma/2.0;
            if gain > *best_split_score {
                *best_split_score = gain;
                *best_threshold = (x_i + x_i_next) / 2.0;
                *best_feature_idx = idx;
            } 
        }
    }
}
#[derive(Clone)]
pub struct XGBRegressorParameters {
    pub n_estimators: usize,
    pub max_depth: u16,
    pub learning_rate: f64,
    pub min_samples_leaf: usize,
    pub lambda: f64,
    pub gamma: f64,
    pub base_score: f64,
    pub subsample: f64,
    pub seed: u64,
}

impl Default for XGBRegressorParameters {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.3,
            max_depth: 6,
            min_samples_leaf: 1,
            lambda: 1.0,
            gamma: 0.0,
            base_score: 0.5,
            subsample: 1.0,
            seed: 0,
        }
    }
}

impl XGBRegressorParameters {
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }
    pub fn with_max_depth(mut self, max_depth: u16) -> Self {
        self.max_depth = max_depth;
        self
    }
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }
    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }
    pub fn with_base_score(mut self, base_score: f64) -> Self {
        self.base_score = base_score;
        self
    }
    pub fn with_subsample(mut self, subsample: f64) -> Self {
        self.subsample = subsample;
        self
    }
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

pub struct XGBRegressor<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>> {
    regressors: Option<Vec<DecisionTreeRegressor<TX, f64, X, Vec<f64>>>>,
    parameters: Option<XGBRegressorParameters>,
    _phantom_ty: PhantomData<TY>,
    _phantom_y: PhantomData<Y>,
}

impl<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>> XGBRegressor<TX, TY, X, Y> {
    /// Fits the XGBoost regressor to the training data.
    ///
    /// This function implements the gradient boosting algorithm. It iteratively fits decision
    /// trees to the residuals of the previous predictions.
    ///
    /// # Arguments
    ///
    /// * `data` - The input feature data.
    /// * `y` - The target values.
    /// * `parameters` - The hyperparameters for the XGBoost model.
    ///
    /// # Returns
    ///
    /// A `Result` containing the trained `XGBoostRegressor` or a `Failed` error.
    pub fn fit(data: &X, y: &Y, parameters: XGBRegressorParameters) -> Result<Self, Failed> {
        // Start with an initial prediction, often the mean of the target values.
        if parameters.subsample > 1.0 {
             return Err(Failed::because(FailedError::ParametersError, &format!(
                "Incorrect subsample ratio: {}. A subsample ratio of less than or equal to 1.0 is required.",
                parameters.subsample
            )));
        }
        let (n_samples, n_features) = data.shape();
        let initial_prediction = parameters.base_score;
        let learning_rate = parameters.learning_rate;

        // Convert the target labels to a Vec<f64>.
        let labels: Vec<f64> = y.iterator(0).map(|v| v.to_f64().unwrap()).collect();

        // The first set of residuals is the difference between the labels and the initial prediction.
        let mut residuals: Vec<f64> = labels
            .iter()
            .map(|label| *label - initial_prediction)
            .collect();

        let mut regressors = Vec::new();
        let decision_tree_params = DecisionTreeRegressorParameters::default()
            .with_max_depth(parameters.max_depth)
            .with_min_samples_leaf(parameters.min_samples_leaf);
        let mut rng = get_rng_impl(Some(parameters.seed));

        // Iteratively build the ensemble of decision trees.
        for _ in 0..parameters.n_estimators {
            // Train a new decision tree on the current residuals.
            let samples = Self::create_subsample_mask(
                n_samples,
                (parameters.subsample * n_samples as f64) as usize,
                &mut rng,
            );

            let decision_tree_regressor = DecisionTreeRegressor::fit_weak_learner(
                data,
                &residuals,
                samples,
                n_features,
                decision_tree_params.clone(),
                Some(parameters.lambda),
                Some(parameters.gamma),
            )
            .unwrap();

            // Get the predictions from the newly trained tree. These are the "predicted residuals".
            let predicted_residuals = decision_tree_regressor.predict(data).unwrap();

            // Update the residuals for the next iteration.
            // Each residual is updated by subtracting the prediction of the new tree, scaled by the learning rate.
            residuals = zip(residuals.iter(), predicted_residuals.iter())
                .map(|(residual, pred_residual)| *residual - (learning_rate * *pred_residual))
                .collect();

            // Add the trained tree to our ensemble.
            regressors.push(decision_tree_regressor);
        }

        // Return the fully trained model.
        Ok(Self {
            regressors: Some(regressors),
            parameters: Some(parameters),
            _phantom_ty: PhantomData,
            _phantom_y: PhantomData,
        })
    }

    pub fn predict(&self, data: &X) -> Result<Vec<TX>, Failed> {
        let (n_samples, _) = data.shape();
        let parameters = self.parameters.as_ref().unwrap();
        let mut initial_predictions = vec![parameters.base_score; n_samples];
        let learning_rate = parameters.learning_rate;
        let regressors = self.regressors.as_ref().unwrap();
        for regressor in regressors.iter() {
            let corrections = regressor.predict(data).unwrap();
            initial_predictions = zip(initial_predictions, corrections)
                .map(|(prediction, correction)| prediction + (learning_rate * correction))
                .collect();
        }
        Ok(initial_predictions
            .into_iter()
            .map(|v| TX::from(v).unwrap())
            .collect())
    }
    fn create_subsample_mask(
        nrows: usize,
        subsample_size: usize,
        rng: &mut impl Rng,
    ) -> Vec<usize> {
        // Ensure we don't try to sample more items than are available.

        // 1. Create a vector containing all possible indices from 0 to nrows-1.
        let mut all_indices: Vec<usize> = (0..nrows).collect();

        // 2. Shuffle the entire list of indices randomly.
        all_indices.shuffle(rng);

        // 3. Create the result vector, initialized with zeros.
        let mut sample_mask = vec![0; nrows];

        // 4. Iterate through the first `subsample_size` shuffled indices
        //    and set the corresponding position in the mask to 1.
        for i in 0..subsample_size {
            let selected_index = all_indices[i];
            sample_mask[selected_index] = 1;
        }

        sample_mask
    }
}

impl<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    SupervisedEstimatorBorrow<'_, X, Y, XGBRegressorParameters> for XGBRegressor<TX, TY, X, Y>
{
    fn new() -> Self {
        Self {
            regressors: None,
            parameters: None,
            _phantom_y: PhantomData,
            _phantom_ty: PhantomData,
        }
    }

    fn fit(x: &X, y: &Y, parameters: &XGBRegressorParameters) -> Result<Self, Failed> {
        XGBRegressor::fit(x, y, parameters.clone())
    }
}

impl<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>> PredictorBorrow<'_, X, TX>
    for XGBRegressor<TX, TY, X, Y>
{
    fn predict(&self, x: &X) -> Result<Vec<TX>, Failed> {
        self.predict(x)
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::linalg::basic::matrix::DenseMatrix;

//     #[test]
//     fn test_fit_initialization() {
//         // Simple dataset
//         let x = DenseMatrix::from_2d_array(&[&[1.0], &[2.0], &[3.0], &[4.0]]).unwrap();
//         let y: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0];

//         let parameters = XGBRegressorParameters {
//             n_estimators: 10,
//             learning_rate: 0.3,
//         };

//         let model =
//             XGBRegressor::<f64, f64, DenseMatrix<f64>, Vec<f64>>::fit(&x, &y, parameters).unwrap();

//         // 1. Test initial prediction
//         let expected_initial_prediction = 25.0; // (10+20+30+40)/4
//         assert_eq!(
//             model.initial_prediction.unwrap(),
//             expected_initial_prediction
//         );

//         // 2. Test number of regressors
//         assert_eq!(model.regressors.as_ref().unwrap().len(), 10);

//         // 3. Test parameters are stored
//         assert_eq!(model.parameters.as_ref().unwrap().n_estimators, 10);
//         assert_eq!(model.parameters.as_ref().unwrap().learning_rate, 0.3);
//     }

//     #[test]
//     fn test_predict_single_estimator() {
//         // Simple dataset where a single split is obvious
//         let x = DenseMatrix::from_2d_array(&[
//             &[1.0],
//             &[2.0], // Low group
//             &[9.0],
//             &[10.0], // High group
//         ])
//         .unwrap();
//         let y: Vec<f64> = vec![10.0, 10.0, 100.0, 100.0];

//         let parameters = XGBRegressorParameters {
//             n_estimators: 1,    // Only one tree
//             learning_rate: 1.0, // No scaling for simplicity
//         };

//         let expected_predictions = vec![10.0, 10.0, 100.0, 100.0];

//         // --- Model Calculation ---
//         let model =
//             XGBRegressor::<f64, f64, DenseMatrix<f64>, Vec<f64>>::fit(&x, &y, parameters).unwrap();
//         let predictions = model.predict(&x).unwrap();

//         // Assert that the model's predictions match the manual calculation
//         for (p, e) in predictions.iter().zip(expected_predictions.iter()) {
//             assert!((p - e).abs() < 1e-9);
//         }
//     }

//     #[test]
//     fn test_predict_multiple_estimators_with_learning_rate() {
//         let x = DenseMatrix::from_2d_array(&[&[1.0], &[2.0], &[9.0], &[10.0]]).unwrap();
//         let y: Vec<f64> = vec![10.0, 10.0, 100.0, 100.0];

//         let parameters = XGBRegressorParameters {
//             n_estimators: 2,
//             learning_rate: 0.5,
//         };

//         // --- Manual Calculation ---
//         // Round 1
//         let initial_pred = 55.0;
//         // Tree 1 predicts r1, outputting -45.0 for low group, +45.0 for high group
//         let pred1_low = initial_pred + 0.5 * (-45.0); // = 32.5;
//         let pred1_high = initial_pred + 0.5 * (45.0); // = 77.5;

//         // Round 2
//         // New residuals (r2 = actual - pred1)
//         let r2_low = 10.0 - pred1_low; // = -22.5;
//         let r2_high = 100.0 - pred1_high; // = 22.5;
//                                           // Tree 2 predicts r2, outputting -22.5 for low group, +22.5 for high group

//         // Final Prediction = pred1 + learning_rate * tree2_prediction
//         let final_pred_low = pred1_low + 0.5 * (r2_low); // = 21.25;
//         let final_pred_high = pred1_high + 0.5 * (r2_high); // = 88.75;
//         let expected_predictions = vec![
//             final_pred_low,
//             final_pred_low,
//             final_pred_high,
//             final_pred_high,
//         ];

//         // --- Model Calculation ---
//         let model =
//             XGBRegressor::<f64, f64, DenseMatrix<f64>, Vec<f64>>::fit(&x, &y, parameters).unwrap();
//         let predictions: Vec<f64> = model.predict(&x).unwrap();

//         for (p, e) in predictions.iter().zip(expected_predictions.iter()) {
//             assert!((p - e).abs() < 1e-9);
//         }
//     }

//     #[test]
//     fn test_predict_on_unseen_data() {
//         let x_train =
//             DenseMatrix::from_2d_array(&[&[1.0], &[2.0], &[3.0], &[10.0], &[11.0], &[12.0]])
//                 .unwrap();
//         let y_train: Vec<f64> = vec![10.0, 10.0, 10.0, 100.0, 100.0, 100.0];

//         let parameters = XGBRegressorParameters {
//             n_estimators: 5,
//             learning_rate: 0.3,
//         };

//         let model = XGBRegressor::<f64, f64, DenseMatrix<f64>, Vec<f64>>::fit(
//             &x_train,
//             &y_train,
//             parameters,
//         )
//         .unwrap();

//         // Test data that falls into the learned categories
//         let x_test = DenseMatrix::from_2d_array(&[
//             &[2.5],  // Should be close to 10
//             &[10.5], // Should be close to 100
//         ])
//         .unwrap();

//         let predictions: Vec<f64> = model.predict(&x_test).unwrap();

//         // Check output shape
//         assert_eq!(predictions.len(), 2);

//         // Check logical correctness
//         // The first prediction should be significantly lower than the second
//         assert!(predictions[0] < predictions[1]);
//         // The first prediction should be closer to 10 than 100
//         assert!((predictions[0] - 10.0).abs() < (predictions[0] - 100.0).abs());
//         // The second prediction should be closer to 100 than 10
//         assert!((predictions[1] - 100.0).abs() < (predictions[1] - 10.0).abs());
//     }
// }
