# Exoplanet Classification with NASA Kepler Data

A machine learning project to classify exoplanets using historical data from NASA's Kepler Space Telescope mission (2009-2019).

## Overview

This project analyzes Kepler Space Telescope data to classify potential exoplanets as either confirmed planets or false positives. While Kepler's mission ended in 2019, its dataset remains one of the most comprehensive resources for exoplanet research.

## Project Structure

```
mine/
├── data/
│   ├── data_collection.py     # NASA Kepler API data extraction
│   └── data_preprocessing.py  # Data cleaning and preparation
├── regressions/
│   └── binary_classification.py  # ML models implementation
├── README.md
└── project_description.txt
```

## Data Pipeline

1. **Data Collection** (`data/data_collection.py`)
   - Fetches historical data from NASA's Kepler API
   - Downloads complete dataset of Kepler observations
   - Run with: `python data/data_collection.py`

2. **Data Preprocessing** (`data/data_preprocessing.py`)
   - Cleans raw Kepler data
   - Handles missing values
   - Standardizes features
   - Creates processed dataset
   - Run with: `python data/data_preprocessing.py`

3. **Machine Learning Models** (`regressions/binary_classification.py`)
   - Implements multiple classifiers:
     * Random Forest
     * Logistic Regression
     * AdaBoost
     * XGBoost
     * K-Nearest Neighbors
   - Run with: `python regressions/binary_classification.py`

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/exoplanet-classification.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Unix/macOS
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Requirements

```
numpy>=1.19.2
pandas>=1.2.3
scikit-learn>=0.24.1
xgboost>=1.3.3
matplotlib>=3.3.4
seaborn>=0.11.1
```

## Usage

1. First, collect the Kepler data:
```bash
python data/data_collection.py
```

2. Preprocess the collected data:
```bash
python data/data_preprocessing.py
```

3. Run the classification models:
```bash
python regressions/binary_classification.py
```

## Results

The project generates:
- Confusion matrices for each model
- Classification reports with precision, recall, and F1-scores
- ROC-AUC scores for model comparison

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NASA Kepler Mission for providing the dataset
- This is my first machine learning project, focused on learning and implementing various preprocessing techniques and ML models
