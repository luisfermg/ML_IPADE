import argparse
import pandas as pd
import joblib

def main():
    parser = argparse.ArgumentParser(description="Generate predictions for new passenger data")
    parser.add_argument("csv", help="Path to input CSV file")
    parser.add_argument("--model", default="best_model.joblib", help="Trained model file")
    args = parser.parse_args()

    model = joblib.load(args.model)
    df = pd.read_csv(args.csv).iloc[:, 1:]
    df = df.drop(columns=["id"], errors="ignore")
    preds = model.predict(df)
    for p in preds:
        print(p)

if __name__ == "__main__":
    main()
