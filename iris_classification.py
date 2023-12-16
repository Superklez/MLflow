import os
import mlflow
import pathlib
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score


def get_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted')
    }


def main():
    # Set and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--penalty', type=str, required=False, default='l2')
    parser.add_argument('--dual', type=bool, required=False, default=False)
    parser.add_argument('--tol', type=float, required=False, default=0.0001)
    parser.add_argument('--C', type=float, required=False, default=1.0)
    parser.add_argument('--fit-intercept', type=bool, required=False,
        default=True)
    parser.add_argument('--solver', type=str, required=False, default='lbfgs')
    parser.add_argument('--max-iter', type=int, required=False, default=100)
    parser.add_argument('--l1-ratio', type=float, required=False,
        default=None)
    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if k != 'args'}
    if args['penalty'] != 'l1':
        del args['l1_ratio']

    # Load data
    dirname = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(dirname, 'data', 'iris.csv')
    df = pd.read_csv(filepath)

    # Split data to x/y and train/test
    x, y = df.iloc[:, :-1], df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,
        random_state=42)
    
    tracking_uri = pathlib.Path(os.path.join(dirname, 'mlruns')).as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.set_experiment('iris_classification')
    with mlflow.start_run(experiment_id=exp.experiment_id):
        # Preprocess data
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # Train model
        model = LogisticRegression(**args, random_state=42)
        model.fit(x_train, y_train)

        # Get predictions and evaluate
        y_pred = model.predict(x_test)
        metrics = get_metrics(y_test, y_pred)
        for metric in metrics:
            print(f'{metric}: {metrics[metric]}')

        # Log with MLflow
        mlflow.log_params(args)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(filepath)
        mlflow.sklearn.log_model(model, 'logistic_regression_model')

    return 0


if __name__ == '__main__':
    main()

