use std::{iter::zip, marker::PhantomData};

use crate::{api::{PredictorBorrow, SupervisedEstimatorBorrow}, error::Failed, linalg::basic::arrays::{Array1, Array2}, numbers::{basenum::Number, realnum::RealNumber}, tree::decision_tree_regressor::{self, DecisionTreeRegressor, DecisionTreeRegressorParameters}};


#[derive(Clone)]
pub struct XGBRegressorParameters {
    pub n_estimators: usize,
    pub learning_rate: f64
}

impl Default for XGBRegressorParameters {
    fn default() -> Self {
        Self {
            n_estimators: 10,
            learning_rate: 0.1
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
}

pub struct XGBRegressor<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>> {
    regressors: Option<Vec<DecisionTreeRegressor<TX, f64, X, Vec<f64>>>>,
    initial_prediction: Option<f64>,
    parameters: Option<XGBRegressorParameters>,
    _phantom_ty: PhantomData<TY>,
    _phantom_y: PhantomData<Y>
} 

impl<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>> XGBRegressor<TX, TY, X, Y> {
        pub fn predict(&self, data: &X) -> Result<Vec<TX>, Failed> {
         let (n_samples, _) = data.shape();
         let mut initial_predictions = vec![self.initial_prediction.unwrap(); n_samples];
         let parameters = self.parameters.as_ref().unwrap();
         let learning_rate = parameters.learning_rate;
         let regressors = self.regressors.as_ref().unwrap();
         for regressor in regressors.iter() {
            let corrections = regressor.predict(data).unwrap();
            println!("{:?}", corrections);
            initial_predictions = zip(initial_predictions, corrections).map(|(prediction, correction)| prediction + (learning_rate * correction)).collect();
         }
         Ok(initial_predictions.into_iter().map(|v| TX::from(v).unwrap()).collect())
    } 
}

impl<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>>
   SupervisedEstimatorBorrow<'_, X, Y, XGBRegressorParameters> for XGBRegressor<TX, TY, X, Y> 
{
    fn new() -> Self {
        Self {
            regressors: None,
            initial_prediction: None,
            parameters: None,
            _phantom_y: PhantomData,
            _phantom_ty: PhantomData
        }
    }

    fn fit(
        x: & X,
        y: &Y,
        parameters: &XGBRegressorParameters,
    ) -> Result<Self, Failed> {
        XGBRegressor::fit(x, y, parameters.clone())
    }
}

impl<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    PredictorBorrow<'_, X, TX> for XGBRegressor<TX, TY, X, Y>
{
    fn predict(&self, x: &X) -> Result<Vec<TX>, Failed> {
        self.predict(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::matrix::DenseMatrix;

   #[test]
    fn test_fit_initialization() {
        // Simple dataset
        let x = DenseMatrix::from_2d_array(&[
            &[1.0], &[2.0], &[3.0], &[4.0]
        ]).unwrap();
        let y: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0];

        let parameters = XGBRegressorParameters {
            n_estimators: 10,
            learning_rate: 0.3,
        };

        let model = XGBRegressor::<f64, f64, DenseMatrix<f64>, Vec<f64>>::fit(&x, &y, parameters).unwrap();

        // 1. Test initial prediction
        let expected_initial_prediction = 25.0; // (10+20+30+40)/4
        assert_eq!(model.initial_prediction.unwrap(), expected_initial_prediction);

        // 2. Test number of regressors
        assert_eq!(model.regressors.as_ref().unwrap().len(), 10);

        // 3. Test parameters are stored
        assert_eq!(model.parameters.as_ref().unwrap().n_estimators, 10);
        assert_eq!(model.parameters.as_ref().unwrap().learning_rate, 0.3);
    }

    #[test]
    fn test_predict_single_estimator() {
        // Simple dataset where a single split is obvious
        let x = DenseMatrix::from_2d_array(&[
            &[1.0], &[2.0], // Low group
            &[9.0], &[10.0] // High group
        ]).unwrap();
        let y: Vec<f64> = vec![10.0, 10.0, 100.0, 100.0];

        let parameters = XGBRegressorParameters {
            n_estimators: 1, // Only one tree
            learning_rate: 1.0, // No scaling for simplicity
        };

        let expected_predictions = vec![10.0, 10.0, 100.0, 100.0];

        // --- Model Calculation ---
        let model = XGBRegressor::<f64, f64, DenseMatrix<f64>, Vec<f64>>::fit(&x, &y, parameters).unwrap();
        let predictions = model.predict(&x).unwrap();

        // Assert that the model's predictions match the manual calculation
        for (p, e) in predictions.iter().zip(expected_predictions.iter()) {
            assert!((p - e).abs() < 1e-9);
        }
    }

    #[test]
    fn test_predict_multiple_estimators_with_learning_rate() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0], &[2.0],
            &[9.0], &[10.0]
        ]).unwrap();
        let y: Vec<f64> = vec![10.0, 10.0, 100.0, 100.0];

        let parameters = XGBRegressorParameters {
            n_estimators: 2,
            learning_rate: 0.5,
        };

        // --- Manual Calculation ---
        // Round 1
        let initial_pred = 55.0;
        // Tree 1 predicts r1, outputting -45.0 for low group, +45.0 for high group
        let pred1_low = initial_pred + 0.5 * (-45.0); // = 32.5;
        let pred1_high = initial_pred + 0.5 * (45.0); // = 77.5;
        
        // Round 2
        // New residuals (r2 = actual - pred1)
        let r2_low = 10.0 - pred1_low; // = -22.5;
        let r2_high = 100.0 - pred1_high; // = 22.5;
        // Tree 2 predicts r2, outputting -22.5 for low group, +22.5 for high group
        
        // Final Prediction = pred1 + learning_rate * tree2_prediction
        let final_pred_low = pred1_low + 0.5 * (r2_low); // = 21.25;
        let final_pred_high = pred1_high + 0.5 * (r2_high); // = 88.75;
        let expected_predictions = vec![final_pred_low, final_pred_low, final_pred_high, final_pred_high];

        // --- Model Calculation ---
        let model = XGBRegressor::<f64, f64, DenseMatrix<f64>, Vec<f64>>::fit(&x, &y, parameters).unwrap();
        let predictions: Vec<f64> = model.predict(&x).unwrap();

        for (p, e) in predictions.iter().zip(expected_predictions.iter()) {
            assert!((p - e).abs() < 1e-9);
        }
    }

    #[test]
    fn test_predict_on_unseen_data() {
        let x_train = DenseMatrix::from_2d_array(&[
            &[1.0], &[2.0], &[3.0],
            &[10.0], &[11.0], &[12.0]
        ]).unwrap();
        let y_train: Vec<f64> = vec![10.0, 10.0, 10.0, 100.0, 100.0, 100.0];

        let parameters = XGBRegressorParameters {
            n_estimators: 5,
            learning_rate: 0.3,
        };

        let model = XGBRegressor::<f64, f64, DenseMatrix<f64>, Vec<f64>>::fit(&x_train, &y_train, parameters).unwrap();

        // Test data that falls into the learned categories
        let x_test = DenseMatrix::from_2d_array(&[
            &[2.5], // Should be close to 10
            &[10.5] // Should be close to 100
        ]).unwrap();

        let predictions: Vec<f64> = model.predict(&x_test).unwrap();

        // Check output shape
        assert_eq!(predictions.len(), 2);

        // Check logical correctness
        // The first prediction should be significantly lower than the second
        assert!(predictions[0] < predictions[1]);
        // The first prediction should be closer to 10 than 100
        assert!((predictions[0] - 10.0).abs() < (predictions[0] - 100.0).abs());
        // The second prediction should be closer to 100 than 10
        assert!((predictions[1] - 100.0).abs() < (predictions[1] - 10.0).abs());
    }
}