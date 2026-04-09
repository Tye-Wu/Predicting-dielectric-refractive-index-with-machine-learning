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

Among the tested models, `HistGradientBoostingRegressor` gave the best overall balance of predictive performance and computational efficiency on this assignment.
