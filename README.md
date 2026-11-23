
# Karnataka Rainfall Prediction Using Hybrid CNN-BiLSTM

## Project Overview

This project develops a deep learning model to predict weekly rainfall
in Karnataka, India, using a hybrid Convolutional Neural Network (CNN)
and Bidirectional Long Short-Term Memory (BiLSTM) architecture.

The project uses historical daily gridded rainfall data from the
India Meteorological Department (IMD), covering 1991–2020.

## Research Area

Sensor Data Analytics

## Objectives

- Analyse historical rainfall patterns across Karnataka.
- Preprocess daily rainfall observations into weekly rainfall data.
- Develop a hybrid CNN-BiLSTM deep learning model.
- Predict the following week's state-wide rainfall.
- Evaluate performance against a persistence baseline.
- Visualise historical rainfall patterns and model performance.

## Dataset

Source: India Meteorological Department (IMD)

Dataset: 0.25-degree daily gridded rainfall data.

Period: 1991–2020.

Geographical coverage: Karnataka, India.

The original NetCDF dataset is not included in this repository.
Refer to the data directory and project documentation for details.

## Methodology

1. NetCDF data inspection.
2. Karnataka geographical boundary extraction.
3. Spatial aggregation and daily rainfall preprocessing.
4. Weekly rainfall aggregation.
5. Exploratory data analysis.
6. Chronological training, validation and test split.
7. MinMax scaling and 12-week sequence preparation.
8. CNN-BiLSTM model development.
9. Model evaluation and visualisation.

## Model Architecture

- Input: 12 previous weeks of rainfall
- Conv1D: 64 filters
- MaxPooling1D
- Bidirectional LSTM: 64 units
- Dropout: 0.2
- Dense: 32 neurons
- Output: 1 neuron

## Test Results

| Metric | CNN-BiLSTM | Persistence |
|---|---:|---:|
| MAE | 10.44 mm | 12.78 mm |
| RMSE | 18.53 mm | 23.89 mm |
| R² | 0.5433 | 0.2410 |

The CNN-BiLSTM model achieved approximately 18.3% lower MAE
than the persistence baseline on the test dataset.

## Project Structure

- data/: Karnataka geographical boundary
- processed/: Prepared rainfall datasets and scaler
- plots/: Exploratory data analysis visualisations
- results/: Trained models, predictions and evaluation results
- inspect_netcdf.py: NetCDF inspection
- phase2_preprocessing.py: Data preprocessing
- phase3_visualization.py: Exploratory visualisation
- phase4_prepare_data.py: Model data preparation

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Xarray
- GeoPandas

## Important Note

This model predicts state-wide weekly rainfall. It does not
produce district-level or grid-level rainfall predictions.

Historical spatial rainfall maps are a separate visualisation
of the original IMD gridded dataset.

## Author

Academic project – MDA507-5
Neural Networks and Deep Learning
November 2026
