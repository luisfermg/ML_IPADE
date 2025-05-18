# ML_IPADE
Repository for IPADE's Machine Learning class

## Dataset

This project uses the **Airline Passenger Satisfaction** dataset available on
[Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction).
The data contains demographic information, service ratings and a target label
indicating whether a passenger was satisfied with their flight experience. The
`train.csv` and `test.csv` files in this repository correspond to splits of the
original dataset and are used by both the script and the notebook.

## Setup

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

## Running the script

Execute the classifier directly from the command line:

```bash
python passenger_classification.py
```

The script prints a few summary statistics followed by the test accuracy. An
example run produces output similar to:

```
Average passenger age: 39.38
Training set satisfaction counts: {'neutral or dissatisfied': 58879, 'satisfied': 45025}
Test accuracy: 0.8863
```

## Running the notebook

Start Jupyter and open the provided notebook:

```bash
jupyter notebook
```

From the Jupyter interface, open `passenger_classification.ipynb` and run the cells to reproduce the results.

The notebook follows these main steps:

1. Load the training data with `pandas`.
2. Split the data into training and test sets.
3. Build a preprocessing and classification pipeline using `ColumnTransformer`
   and `LogisticRegression` from scikit-learn.
4. Fit the model and evaluate it with accuracy and a classification report.

Executing each cell sequentially will reproduce the metrics shown in the
notebook.
