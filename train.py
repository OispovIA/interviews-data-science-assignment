import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool

def load_data(path):
    data = pd.read_csv(path)
    num_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']
    data = data[(data[num_cols] > 0).all(axis=1)]
    return data

def split_and_save_data(data, train_save_path=None, test_save_path=None):
    cat_cols = ['cut', 'color', 'clarity']
    target_col = ['price']

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=target_col),
        data[target_col],
        test_size=0.3,
        random_state=42
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    test_pool = Pool(X_test, y_test, cat_features=cat_cols)
    
    if train_save_path and test_save_path:
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        train_data.to_csv(train_save_path, index=False)
        test_data.to_csv(test_save_path, index=False)
    
    return train_pool, test_pool, X_test

def train_model(train_pool, depth, learning_rate, loss_function, iterations, task_type, verbose):
    catboost = CatBoostRegressor(
        depth=depth, 
        learning_rate=learning_rate, 
        loss_function=loss_function,
        iterations=iterations,
        task_type=task_type,
        verbose=verbose
    )
    catboost.fit(train_pool)
    return catboost

def save_model(model, path):
    if path is not None:
        model.save_model(path)

def load_model(path):
    if path is not None:
        return CatBoostRegressor().load_model(path)

def save_predictions(model, test_pool, X_test, path, format):
    if path is not None:
        predictions = model.predict(test_pool)
        output = pd.DataFrame({'Id': X_test.index, 'Prediction': predictions})
        if format == 'csv':
            output.to_csv(path, index=False)
        elif format == 'excel':
            output.to_excel(path, index=False)

def main(args):
    data = load_data(args.data_path)
    train_pool, test_pool, X_test = split_and_save_data(data, args.train_save_path, args.test_save_path)

    if args.load_model_path:
        model = load_model(args.load_model_path)
    else:
        model = train_model(
            train_pool, 
            args.depth, 
            args.learning_rate, 
            args.loss_function, 
            args.iterations,
            args.task_type, 
            args.verbose
        )

    score = model.score(test_pool)
    print(f"Model R2 Score: {score}")

    save_predictions(model, test_pool, X_test, args.output_path, args.format)
    save_model(model, args.save_model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model on diamond data and make predictions")

    parser.add_argument('--data_path', type=str, required=True, help='Path to the diamonds.csv file')
    
    parser.add_argument('--depth', type=int, default=8, help='Depth for CatBoost regressor')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate for CatBoost regressor')
    parser.add_argument('--loss_function', type=str, default='RMSE', help='Loss function for CatBoost regressor')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations for CatBoost regressor')
    parser.add_argument('--verbose', type=int, default=200, help='Verbose value for CatBoost regressor')
    parser.add_argument('--task_type', type=str, default='CPU', choices=['CPU', 'GPU'], help='Task type for CatBoost')
    
    parser.add_argument('--train_save_path', type=str, default=None, help='Optional path to save the training data as CSV')
    parser.add_argument('--test_save_path', type=str, default=None, help='Optional path to save the testing data as CSV')
    
    parser.add_argument('--output_path', type=str, default=None, help='Path to save the predictions')
    parser.add_argument('--format', type=str, choices=['csv', 'excel'], default='csv', help='Format for saving predictions')
    
    parser.add_argument('--save_model_path', type=str, default=None, help='Path to save the trained model')
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to load a pre-trained model')

    args = parser.parse_args()
    main(args)