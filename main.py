import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from local.constants import TMDB_AUTHORIZATION
from movie_api import MovieAPI


def load_data():
    train_data = pd.read_csv('movie/train.csv', sep=';', header=None)
    movies = pd.read_csv('movie/movie.csv', sep=';', header=None)
    tmdb_ids = movies.iloc[:, 1]
    movie_ids = movies.iloc[:, 0]
    return train_data, tmdb_ids, movie_ids


def euclidean_distance(X, Y):
    if len(X) != len(Y):
        raise ValueError("Not same number of features to compute euclidean distance")
    return sum((x - y) ** 2 for x, y in zip(X, Y))


def jaccard_distance(X, Y):
    intersection = len(X.intersection(Y))
    union = len(X.union(Y))
    jaccard_similarity = intersection / union
    return 1 - jaccard_similarity


def fetch_movie_details(tmdb_ids, movie_ids, api_key):
    movie_api = MovieAPI(api_key=api_key)
    movie_details_df = movie_api.fetch_multiple_movies(tmdb_ids, movie_ids)
    return movie_details_df


def knn1(user_id, train_set, validation_set, movie_details_dict):
    filter_set = validation_set[validation_set.iloc[:, 1] == user_id]
    validation_movie_ids = filter_set.iloc[:, 2].values
    validation_ids = filter_set.iloc[:, 0].values
    for validation_movie_id, validation_id in zip(validation_movie_ids, validation_ids):
        print(validation_movie_id, validation_id)
        knn2(user_id, train_set, validation_movie_id, validation_id, movie_details_dict)


def knn2(user_id, train_set, validation_movie_id, validation_id, movie_details_df):
    filter_data = train_set[train_set.iloc[:, 1] == user_id]
    train_movie_ids = filter_data.iloc[:, 2]
    train_ratings = filter_data.iloc[:, 3]

    for train_movie_id in train_movie_ids:
        total_distance(train_movie_id, validation_movie_id, movie_details_df)
        # compare(validation_movie_id, train_movie_id)


def compute_numerical_distance(train_movie, validation_movie, numerical_features):
    train_features = train_movie[numerical_features].to_numpy()
    validation_features = validation_movie[numerical_features].to_numpy()
    return euclidean_distance(train_features, validation_features)

def compute_categorical_distance(train_movie, validation_movie, categorical_features):
    train_features = train_movie[categorical_features].to_numpy()
    validation_features = validation_movie[categorical_features].to_numpy()
    distance = 0
    for categorical_feature in categorical_features:
        distance += euclidean_distance()
    return jaccard_distance(train_features, validation_features)

def total_distance(train_movie_id, validation_movie_id, movie_details_df):
    train_movie = movie_details_df[movie_details_df["movie_id"] == train_movie_id].iloc[0]
    validation_movie = movie_details_df[movie_details_df["movie_id"] == validation_movie_id].iloc[0]

    numerical_features = ["vote_average", "vote_count"]
    numerical_distance = compute_numerical_distance(train_movie, validation_movie, numerical_features)

    categorical_features = ["genere"]
    categorical_distance = compute_categorical_distance()

    distance = numerical_distance
    print(f"train movie type: {type(train_movie)}")
    print(f"distance: {distance}")
    return distance


def normalize(df):
    numeric_columns = ['vote_average', 'vote_count']
    if not all(col in df.columns for col in numeric_columns):
        raise ValueError("DataFrame must contain 'vote_average' and 'vote_count' columns")
    scaler = MinMaxScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df


def main():
    # Load and prepare data
    train_data, tmdb_ids, movie_ids = load_data()
    train_set, validation_set = train_test_split(train_data, test_size=0.2, random_state=42)
    movie_details_df = fetch_movie_details(tmdb_ids, movie_ids, TMDB_AUTHORIZATION)
    movie_details_df = normalize(movie_details_df)

    user_ids = train_data.iloc[:, 1].unique()
    for user_id in user_ids:
        knn1(user_id, train_set, validation_set, movie_details_df)
        break  # todo Delate later, only for testing
    # for user_id in validation_set.:
    #     knn(user_id, train_set, movie_id, movie_details_dd)


if __name__ == '__main__':
    main()
