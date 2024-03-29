import os
import mlflow
import pathlib
import argparse
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec


def get_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted')
    }


def main():
    # Set arguments
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
    parser.add_argument('--save-model', type=bool, required=False,
                        default=False)
    parser.add_argument('--max-input-examples', type=int, required=False,
                        default=3)

    # Parse arguments
    args = parser.parse_args()
    params = {k: v for k, v in vars(args).items() \
              if k not in ('args', 'save_model', 'max_input_examples')}
    if params['penalty'] != 'l1':
        del params['l1_ratio']

    # Load data
    dirpath = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(dirpath, 'data', 'iris.csv')
    df = pd.read_csv(filepath)

    # Split data to x/y and train/test
    x, y = df.iloc[:, :-1], df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,
                                                        random_state=42)
    
    # Start MLflow experiment
    tracking_uri = pathlib.Path(os.path.join(dirpath, 'mlruns')).as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.set_experiment('iris_classification')

    with mlflow.start_run(experiment_id=exp.experiment_id):
        # Create and train pipeline
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(**params, random_state=42))
        ])
        pipe.fit(x_train, y_train)

        # Get predictions and evaluate
        y_pred = pipe.predict(x_test)
        metrics = get_metrics(y_test, y_pred)
        for metric in metrics:
            print(f'{metric}: {metrics[metric]}')

        # Define model signatures and input examples
        input_schema = Schema([
            ColSpec('double', 'sepal_length'),
            ColSpec('double', 'sepal_width'),
            ColSpec('double', 'petal_length'),
            ColSpec('double', 'petal_width')
        ])
        output_schema = Schema([ColSpec('string')])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        input_examples = x.iloc[:args.max_input_examples, :]

        # Log with MLflow
        pipe_name = 'iris_classification_pipeline'
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(filepath)
        mlflow.sklearn.log_model(pipe, pipe_name, signature=signature,
                                 input_example=input_examples)
        if args.save_model:
            pipe_dirpath = os.path.join(dirpath, 'mlmodels', pipe_name)
            mlflow.sklearn.save_model(pipe, pipe_dirpath,
                                      signature=signature,
                                      input_example=input_examples)

    return 0


if __name__ == '__main__':
    main()

