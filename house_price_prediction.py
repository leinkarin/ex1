import pandas as pd
import numpy as np
import os

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime
from typing import NoReturn
from linear_regression import LinearRegression


def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    preprocess training df.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded df
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the df
    """
    # Combine X and y into a single DataFrame
    df = X.copy()  # Use copy to avoid modifying X directly
    if y is not None:
        df['price'] = y

    df = df.dropna().drop_duplicates()

    # Dropping redundant features
    df = df.drop(columns=['id', 'date'])

    # Removing rows where the features are less than zero
    df = df[df['sqft_above'] >= 0]
    df = df[df['sqft_basement'] >= 0]
    df = df[df['sqft_living15'] >= 0]
    df = df[df['sqft_living'] >= 0]
    df = df[df['bedrooms'] >= 0]
    df = df[df['bathrooms'] >= 0]
    df = df[df['floors'] >= 0]
    df = df[df['yr_built'] >= 0]
    df = df[df['yr_renovated'] >= 0]

    # Removing rows where the features are equal to zero or less
    df = df[df['sqft_lot15'] > 0]
    df = df[df['sqft_lot'] > 0]

    # Removing rows where the features are not in the specified range
    df = df[df['waterfront'].isin([0, 1])]
    df = df[df['view'].isin(range(5))]  # 0-4
    df = df[df['condition'].isin(range(1, 6))]  # 1-5
    df = df[df['grade'].isin(range(1, 14))]  # 1-13

    # Removing rows where year built is greater than year renovated, only if yr_renovated is non-zero
    df = df[(df['yr_renovated'] == 0) | (df['yr_built'] <= df['yr_renovated'])]

    # Adding features
    current_year = datetime.now().year
    df['house_age'] = df['yr_built'].apply(lambda x: current_year - x)
    df['is_renovated'] = df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)

    y_processed = df['price']
    X_processed = df.drop(columns=['price'])

    return X_processed, y_processed


def preprocess_test(X: pd.DataFrame):
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """

    df = X.copy()  # Use copy to avoid modifying X directly

    # Dropping redundant features
    df = df.drop(columns=['id', 'date'])

    # Adding features
    current_year = datetime.now().year
    df['house_age'] = df['yr_built'].apply(lambda x: current_year - x)
    df['is_renovated'] = df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)

    return df


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for feature in X.columns:
        # Calculate Pearson correlation coefficient by the formula cov(X, Y) / (std(X) * std(Y))
        covariance = np.cov(X[feature], y)[0, 1]
        X_std = np.std(X[feature])
        y_std = np.std(y)
        rho = covariance / (X_std * y_std)

        plt.figure()
        plt.scatter(X[feature], y, alpha=0.5)
        plt.title(f"Correlation Between {feature} Values and Response\nPearson Correlation: {rho:.2f}")
        plt.xlabel(f"{feature} Values")
        plt.ylabel("Response Values")
        plt.savefig(f"{output_path}/{feature}.png")
        plt.close()


def question_6(X_train, X_test, y_train, y_test, output_path="."):
    """
    Fit model over increasing percentages of the overall training data
    @param X_train: pd.DataFrame
    @param X_test: pd.DataFrame
    @param y_train: pd.Series
    @param y_test: pd.Series
    @param output_path: output path to save the plot
    """
    percentages = np.arange(10, 101)  # 10%, 11%, ..., 100%
    mean_losses = []
    std_losses = []

    for p in percentages:
        losses = []
        for i in range(10):
            # Sample p% of the overall training data
            X_sampled = X_train.sample(frac=p / 100)
            y_sampled = y_train.loc[X_sampled.index]  # sample the corresponding y values

            # Fit linear model (including intercept) over sampled set
            model = LinearRegression(include_intercept=True)
            model.fit(X_sampled, y_sampled)

            # Store average and variance of loss over test set
            loss = model.loss(X_test.values, y_test)
            losses.append(loss)

        mean_losses.append(np.mean(losses))
        std_losses.append(np.std(losses))

    mean_losses = np.array(mean_losses)
    std_losses = np.array(std_losses)

    upper_bound = mean_losses + 2 * std_losses
    lower_bound = mean_losses - 2 * std_losses
    plot_question_6(mean_losses, upper_bound, lower_bound, output_path)


def plot_question_6(mean_losses, upper_bound, lower_bound, output_path):
    """
    Plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    @param mean_losses: np.array
    @param upper_bound: np.array
    @param lower_bound: np.array
    @param output_path: output path to save the plot
    """
    percentages = np.arange(10, 101)

    plt.figure(figsize=(10, 6))
    plt.plot(percentages, mean_losses, marker='o', label='Mean Loss')
    plt.fill_between(percentages, lower_bound, upper_bound, color='skyblue', alpha=0.4)

    plt.title('Mean Loss vs. Training Set Percentage')
    plt.xlabel('Training Set Percentage')
    plt.ylabel('Mean Loss')
    plt.legend()
    plt.grid(True)

    file_path = os.path.join(output_path, "mean_loss_vs_training_percentage.png")
    plt.savefig(file_path)
    plt.close()


if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")
    X, y = df.drop("price", axis=1), df.price

    # Question 2 - split train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Question 3 - preprocessing of housing prices train dataset
    X_train, y_train = preprocess_train(X_train, y_train)

    # Question 4 - Feature evaluation of train dataset with respect to response
    current_directory = os.getcwd()
    output_path = os.path.join(current_directory, "output")
    feature_evaluation(X_train, y_train, output_path=output_path)

    # Question 5 - preprocess the test data
    X_test = preprocess_test(X_test)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Question 6 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    question_6(X_train, X_test, y_train, y_test, output_path=output_path)
