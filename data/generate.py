
import argparse
import sklearn.datasets
import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, help="output file path", type=str)
    args = parser.parse_args()

    X,y = sklearn.datasets.make_classification(
            n_samples=1000,
            n_features=10,
            n_classes=2,
        )
    df = pd.DataFrame(X)
    df['Y'] = y
    df.to_csv(args.output, index=False)
