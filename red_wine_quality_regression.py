import os
import mlflow
import pathlib
import argparse
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec


def get_metrics(y_true, y_pred):
    return {
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred, squared=True),
        'rmse': mean_squared_error(y_true, y_pred, squared=False)
    }


def main():
    # Set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, required=False, default=1.0)
    parser.add_argument('--l1-ratio', type=float, required=False, default=0.5)
    parser.add_argument('--fit-intercept', type=bool, required=False,
                        default=True)
    parser.add_argument('--max-iter', type=int, required=False, default=1000)
    parser.add_argument('--tol', type=float, required=False, default=0.0001)
    parser.add_argument('--positive', type=bool, required=False,
                        default=False)
    parser.add_argument('--selection', type=str, required=False,
                        default='cyclic')
    parser.add_argument('--save-model', type=bool, required=False,
                        default=False)
    parser.add_argument('--max-input-examples', type=int, required=False,
                        default=3)

    # Parse arguments
    args = parser.parse_args()
    params = {k: v for k, v in vars(args).items() \
              if k not in ('args', 'save_model', 'max_input_examples')}

    # Load data
    dirpath = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(dirpath, 'data', 'red-wine-quality.csv')
    df = pd.read_csv(filepath)

    # Split data to x/y and train/test
    x, y = df.iloc[:, :-1], df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,
                                                        random_state=42)

    # Start MLflow experiment
    tracking_uri = pathlib.Path(os.path.join(dirpath, 'mlruns')).as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.set_experiment('red_wine_quality_regression')

    with mlflow.start_run(experiment_id=exp.experiment_id):
        # Create and train pipeline
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', ElasticNet(**params, random_state=42))
        ])
        pipe.fit(x_train, y_train)

        # Get predictions and evaluate
        y_pred = pipe.predict(x_test)
        metrics = get_metrics(y_test, y_pred)
        for metric in metrics:
            print(f'{metric}: {metrics[metric]}')

        # Define model signatures and input examples
        input_schema = Schema([
            ColSpec('double', 'fixed_acidity'),
            ColSpec('double', 'volatile_acidity'),
            ColSpec('double', 'citric_acid'),
            ColSpec('double', 'residual_sugar'),
            ColSpec('double', 'chlorides'),
            ColSpec('double', 'free_sulfur_dioxide'),
            ColSpec('double', 'total_sulfur_dioxide'),
            ColSpec('double', 'density'),
            ColSpec('double', 'ph'),
            ColSpec('double', 'sulphates'),
            ColSpec('double', 'alcohol'),
        ])
        output_schema = Schema([ColSpec('double')])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        input_examples = x.iloc[:args.max_input_examples, :]

        # Log with MLflow
        pipe_name = 'red_wine_quality_regression_pipeline'
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipe, pipe_name, signature=signature,
                                 input_example=input_examples)
        if args.save_model:
            pipe_filepath = os.path.join(dirpath, 'mlmodels', pipe_name)
            mlflow.sklearn.save_model(pipe, pipe_filepath,
                                      signature=signature,
                                      input_example=input_examples)

    
    return 0


if __name__ == '__main__':
    main()

