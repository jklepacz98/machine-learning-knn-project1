import os.path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from movie_api import MovieAPI


def load_data():
    train_csv = pd.read_csv('movie/train.csv', sep=';', header=None)
    train_csv.columns = ['id', 'user_id', 'movie_id', "rating"]
    movie_csv = pd.read_csv('movie/movie.csv', sep=';', header=None)
    movie_csv.columns = ['id', 'tmdb_id', 'title']
    return train_csv, movie_csv


def euclidean_distance(X, Y):
    if len(X) != len(Y):
        raise ValueError("Not same number of features to compute euclidean distance")
    return sum((x - y) ** 2 for x, y in zip(X, Y))


def jaccard_distance(X, Y):
    intersection = len(X.intersection(Y))
    union = len(X.union(Y))
    jaccard_similarity = intersection / union
    return 1 - jaccard_similarity


def fetch_movie_details(movie_csv):
    movie_api = MovieAPI()
    movie_details_df = movie_api.fetch_movies(movie_csv)
    return movie_details_df


def calculate_distances(training_set, validation_set):
    distance_data = []
    distances_and_ratings = []
    for _, x in validation_set.iterrows():
        for _, y in training_set.iterrows():
            distance = total_distance(x, y)
            rating = y["rating"]
            distances_and_ratings.append((distance, rating))
        distances_and_ratings = sorted(distances_and_ratings)
        distance_data.append(distances_and_ratings)
    return distance_data


def find_best_k(distance_data, validation_set):
    best_k = None
    best_mae = float('inf')
    correct_predictions_percentage = 0  # To store the percentage of good predictions

    for k in range(1, 40):
        total_mae, correct_predictions, almost_correct_prediction = knn3(k, distance_data, validation_set)

        avg_mae = total_mae / len(validation_set)
        good_predictions_percentage = (correct_predictions / len(validation_set)) * 100
        almost_correct_predictions_percentage = (almost_correct_prediction / len(validation_set)) * 100

        # Update the best k if current k gives a lower average MAE
        if avg_mae < best_mae:
            best_k = k
            best_mae = avg_mae
            correct_predictions_percentage = good_predictions_percentage
    print(f"Best k: {best_k}, "
          f"Best MAE: {best_mae:.3f}, "
          f"Good Predictions Percentage: {correct_predictions_percentage:.2f}%, "
          f"Almost Correct Predictions Percentage: {almost_correct_predictions_percentage:.2f}%")
    return best_k, correct_predictions_percentage, almost_correct_predictions_percentage


def knn2(training_set, validation_set):
    distance_data = calculate_distances(training_set, validation_set)
    best_k, correct_predictions_percentage, almost_correct_predictions_percentage = find_best_k(distance_data, validation_set)
    return best_k, correct_predictions_percentage, almost_correct_predictions_percentage


def knn(user_id, train_csv, movie_csv):
    user_trains = train_csv[train_csv.iloc[:, 1] == user_id]
    user_movies = pd.merge(user_trains, movie_csv, left_on=["movie_id"], right_on=["id"])

    validation_size = int(len(user_movies) * 0.2)
    validation_set = user_movies.iloc[:validation_size]
    training_set = user_movies.iloc[validation_size:]

    return knn2(training_set, validation_set)

def knn3(k, distance_data, validation_set):
    total_mae = 0
    correct_predictions = 0
    almost_correct_prediction = 0
    for idx, x in enumerate(validation_set.iterrows()):
        true_rating = x[1]["rating"]
        distances_and_ratings = distance_data[idx]
        nearest_neighbors = sorted(distances_and_ratings)[:k]
        predicted_rating = np.mean([rating for _, rating in nearest_neighbors])
        rounded_predicted_rating = round(predicted_rating)
        mae = abs(true_rating - predicted_rating)
        total_mae += mae
        if rounded_predicted_rating == true_rating:
            correct_predictions += 1
        if abs(rounded_predicted_rating - true_rating) <= 1:
            almost_correct_prediction += 1
    return total_mae, correct_predictions, almost_correct_prediction


def compute_numerical_distance(x, y, numerical_features):
    x_features = x[numerical_features].to_numpy()
    y_features = y[numerical_features].to_numpy()
    return euclidean_distance(x_features, y_features)


def compute_categorical_distance(x, y, categorical_features):
    distance = 0
    for categorical_feature in categorical_features:
        x_feature = set(x[categorical_feature])
        y_feature = set(y[categorical_feature])
        distance += jaccard_distance(x_feature, y_feature)
    return distance


def total_distance(x, y):
    numerical_features = ["vote_average"]
    numerical_distance = compute_numerical_distance(x, y, numerical_features)

    categorical_features = ["genres"]
    categorical_distance = compute_categorical_distance(x, y, categorical_features)

    distance = categorical_distance + numerical_distance

    return distance


def normalize(df):
    numeric_columns = ['vote_average', 'vote_count']
    if not all(col in df.columns for col in numeric_columns):
        raise ValueError("DataFrame must contain 'vote_average' and 'vote_count' columns")
    scaler = MinMaxScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df


def main():
    train_csv, movie_csv = load_data()
    movie_df = fetch_movie_details(movie_csv)
    movie_df = normalize(movie_df)

    user_ids = train_csv['user_id'].unique()
    accuracy = []
    almost_accuracy = []
    for user_id in user_ids:
        best_k, acc, acc1 = knn(user_id, train_csv, movie_df)
        accuracy.append(acc)
        almost_accuracy.append(acc1)

    print(f"średnia dokładność {sum(accuracy) / len(accuracy)}")
    print(f"średnia prawie dokładność {sum(almost_accuracy) / len(almost_accuracy)}")


if __name__ == '__main__':
    main()
