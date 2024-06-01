import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from polynomial_fitting import PolynomialFitting


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])  # Load dataset and parse 'Date' column as datetime
    df = df.dropna()  # Drop rows with missing values
    df = df.drop_duplicates()  # Drop duplicate rows
    df = df[df['Month'] >= 1]
    df = df[df['Month'] <= 12]  # Drop rows with invalid 'Month' values
    df = df[df['Day'] >= 1]
    df = df[df['Day'] <= 31]  # Drop rows with invalid 'Day' values
    df = df[df['Temp'] >= 0]  # Drop rows with negative temperature values

    # Add 'DayOfYear' column based on the 'Date' column
    df['DayOfYear'] = df['Date'].dt.dayofyear
    return df


def question_3(df_israel):
    temp = df_israel['Temp']
    day_of_year = df_israel['DayOfYear']
    year = df_israel['Year']

    unique_years = year.unique()
    cmap = plt.get_cmap('viridis', len(unique_years))

    # Create a scatter plot for each year with distinct colors
    for i, unique_year in enumerate(unique_years):
        year_mask = (year == unique_year)
        plt.scatter(day_of_year[year_mask], temp[year_mask], label=str(unique_year), color=cmap(i), alpha=0.6)

    plt.xlabel('Day of Year')
    plt.ylabel('Temperature (°C)')
    plt.title('Average Daily Temperature in Israel as a Function of Day of Year')
    plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
    plt.show()

    # Group by 'Month' and calculate the standard deviation of 'Temp'
    monthly_std = df_israel.groupby('Month').agg(Std=('Temp', 'std'))

    # Plot the standard deviation as a bar plot
    plt.figure(figsize=(12, 6))
    monthly_std.plot(kind='bar', color='skyblue')
    plt.xlabel('Month')
    plt.ylabel('Standard Deviation of Temperature (°C)')
    plt.title('Monthly Standard Deviation of Daily Temperatures')
    plt.xticks(rotation=0)
    plt.show()


def question_4(df):
    # Group by 'Country' and 'Month' and calculate average and standard deviation of temperature
    country_month = df.groupby(['Country', 'Month'])['Temp'].agg(['mean', 'std']).reset_index()

    # Plot the standard deviation as a bar plot
    plt.figure(figsize=(12, 6))
    countries = country_month['Country'].unique()
    for country in countries:
        country_data = country_month[country_month['Country'] == country]
        plt.errorbar(country_data['Month'], country_data['mean'], yerr=country_data['std'], label=country, capsize=5)

    plt.title('Average Monthly Temperature by Country with Standard Deviation')
    plt.xlabel('Month')
    plt.ylabel('Temperature (°C)')
    plt.legend(title='Country')
    plt.show()


def question_5(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    losses = {}
    for k in range(1, 11):
        model = PolynomialFitting(k)
        model.fit(X_train, y_train)
        # Calculate the mean squared error and round to 2 decimal places
        mse = model.loss(X_test, y_test)
        losses[k] = round(mse, 2)

    # Print the test errors
    for k, loss in losses.items():
        print(f"Degree {k}: Test Error = {loss}")

    # Plot the test errors
    plt.figure(figsize=(10, 6))
    plt.bar(losses.keys(), losses.values(), color='blue')

    plt.xlabel('Polynomial Degree (k)')
    plt.ylabel('Test Error')
    plt.title('Test Error for Polynomial Models of Different Degrees')
    plt.xticks(range(1, 11))
    plt.show()


def question_6(df, X, y):
    model = PolynomialFitting(7)
    model.fit(X, y)

    countries = df['Country'].unique()
    countries = countries[countries != 'Israel']

    losses = {}
    for country in countries:
        df_country = df[df['Country'] == country]
        X_country = df_country['DayOfYear'].values
        y_country = df_country['Temp'].values
        loss = model.loss(X_country, y_country)
        losses[country] = round(loss, 2)

    # Plot the test errors
    plt.figure(figsize=(10, 6))
    bars = plt.bar(losses.keys(), losses.values(), color='blue')
    # Add values above the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')

    plt.ylabel('Test Error')
    plt.title('Test Error for Country with model for Israel')
    plt.show()


if __name__ == '__main__':
    # Question 2 - Load and preprocessing of city temperature dataset
    df = load_data("city_temperature.csv")
    df_israel = df[df['Country'] == 'Israel']
    # Question 3 - Exploring data for specific country
    question_3(df_israel)

    # Question 4 - Exploring differences between countries
    question_4(df)

    # Question 5 - Fitting model for different values of `k`
    y = df_israel['Temp'].values
    X = df_israel['DayOfYear'].values
    question_5(X, y)

    # Question 6 - Evaluating fitted model on different countries
    question_6(df, X, y)
