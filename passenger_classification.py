# This script trains a simple Naive Bayes classifier on the passenger satisfaction dataset.
# It avoids external dependencies such as numpy or pandas by using only Python's
# standard library. The goal is to provide a minimal example that follows the
# project guidelines despite the limited environment.

import csv
import math
from collections import defaultdict, Counter

# Columns treated as numerical features for the Gaussian part of the model
NUMERIC_COLS = [
    "Age",
    "Flight Distance",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]

# Columns treated as categorical/ordinal discrete features
CATEGORICAL_COLS = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
]

def load_dataset(path):
    """Load a CSV file into a list of dictionaries."""
    data = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Drop the unnamed first column and the id column
            row.pop("", None)
            row.pop("id", None)
            # Replace missing arrival delay with 0
            if row["Arrival Delay in Minutes"] == "":
                row["Arrival Delay in Minutes"] = "0"
            data.append(row)
    return data

class NaiveBayes:
    """Minimal implementation of a mixed (Gaussian + categorical) Naive Bayes."""

    def __init__(self):
        # Prior counts for each class
        self.class_counts = Counter()
        # For numeric features: {class: {col: (mean, var)}}
        self.numeric_stats = defaultdict(dict)
        # For categorical features: {class: {col: Counter}}
        self.cat_counts = defaultdict(lambda: defaultdict(Counter))
        self.total_samples = 0

    def fit(self, data):
        """Estimate parameters from the training data."""
        # First pass to compute sums for numeric features
        sums = defaultdict(lambda: defaultdict(float))
        sums_sq = defaultdict(lambda: defaultdict(float))
        for row in data:
            cls = row["satisfaction"]
            self.class_counts[cls] += 1
            self.total_samples += 1
            # Accumulate stats for numeric columns
            for col in NUMERIC_COLS:
                val = float(row[col])
                sums[cls][col] += val
                sums_sq[cls][col] += val * val
            # Count categorical values
            for col in CATEGORICAL_COLS:
                self.cat_counts[cls][col][row[col]] += 1
        # Compute means and variances
        for cls in self.class_counts:
            for col in NUMERIC_COLS:
                n = self.class_counts[cls]
                mean = sums[cls][col] / n
                var = sums_sq[cls][col] / n - mean * mean
                if var <= 0:
                    var = 1e-6  # avoid zero variance
                self.numeric_stats[cls][col] = (mean, var)

    def _gaussian_log_prob(self, x, mean, var):
        """Log probability of x under a Gaussian distribution."""
        return -0.5 * (math.log(2 * math.pi * var) + (x - mean) ** 2 / var)

    def predict(self, row):
        """Return the predicted class for a single row."""
        best_cls = None
        best_logp = -float("inf")
        for cls in self.class_counts:
            # Start with the log prior
            logp = math.log(self.class_counts[cls] / self.total_samples)
            # Numeric features
            for col in NUMERIC_COLS:
                x = float(row[col])
                mean, var = self.numeric_stats[cls][col]
                logp += self._gaussian_log_prob(x, mean, var)
            # Categorical features with Laplace smoothing
            for col in CATEGORICAL_COLS:
                counts = self.cat_counts[cls][col]
                total = sum(counts.values())
                value = row[col]
                logp += math.log((counts[value] + 1) / (total + len(counts)))
            if logp > best_logp:
                best_logp = logp
                best_cls = cls
        return best_cls

    def score(self, data):
        """Compute classification accuracy on a dataset."""
        correct = 0
        for row in data:
            if self.predict(row) == row["satisfaction"]:
                correct += 1
        return correct / len(data)

def main():
    # Load datasets
    train_data = load_dataset("train.csv")
    test_data = load_dataset("test.csv")

    # Compute basic descriptive statistics from the training data
    ages = [int(row["Age"]) for row in train_data]
    avg_age = sum(ages) / len(ages)
    satisfaction_counts = Counter(row["satisfaction"] for row in train_data)
    print(f"Average passenger age: {avg_age:.2f}")
    print("Training set satisfaction counts:", dict(satisfaction_counts))

    # Train the Naive Bayes classifier
    model = NaiveBayes()
    model.fit(train_data)

    # Evaluate on the test dataset
    accuracy = model.score(test_data)
    print(f"Test accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
