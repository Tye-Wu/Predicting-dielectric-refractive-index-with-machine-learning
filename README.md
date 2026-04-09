# Predicting-dielectric-refractive-index-with-machine-learning
# MatBench Dielectric: Refractive Index Prediction

This repository contains a small materials informatics project based on the `matbench_dielectric` dataset from [MatBench](https://matbench.materialsproject.org/). The goal is to predict the refractive index of dielectric materials using structure- and composition-derived descriptors together with classical machine learning models.

## Project overview

The workflow includes:

- loading the `matbench_dielectric` dataset
- preprocessing the target variable using a log transform
- generating descriptors with `matminer` and `pymatgen`
- training and comparing several regression models
- evaluating performance using cross-validation and a held-out test set

## Models used

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- HistGradientBoosting Regressor

## Main result

Non-linear ensemble methods significantly outperform the linear baseline, indicating that the relationship between descriptors and refractive index is not purely linear.

In this study, **Random Forest achieves the best performance on the held-out test set**, slightly outperforming boosting-based methods. This may reflect the relatively noisy, heterogeneous nature of the feature space, where bagging-based methods can provide stronger robustness.

Overall, ensemble tree-based models consistently outperform linear models, highlighting the importance of capturing non-linear structure–property relationships in materials datasets.
