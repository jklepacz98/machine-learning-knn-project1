import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from collections import Counter
import re
import spacy
from local.constants import TMDB_AUTHORIZATION

def main():
    # Read movie data
    train_data = pd.read_csv('movie/train.csv', header=None)
    movies = pd.read_csv('movie/movie.csv', sep=';', header=None)
    movie_ids = movies.iloc[:, 1]

    # Split data into train and validation sets
    train_set, validation_set = train_test_split(train_data, test_size=0.2, random_state=42)

    print(f"Training set size: {len(train_set)}")
    print(f"Validation set size: {len(validation_set)}")

    print(f"Movie IDs: {movie_ids}")

    nlp = spacy.load("en_core_web_sm")

    # API headers for movie details
    headers = {
        "accept": "application/json",
        "Authorization": TMDB_AUTHORIZATION
    }

    movie_details = {}
    overviews = []  # List to store all overviews
    genres_list = []  # List to store all genres
    actors_list = []  # List to store all actors

    for movie_id in movie_ids:
        # Get basic movie details
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            movie_data = response.json()
            movie_details[movie_id] = movie_data  # Store each movie's details by ID

            # Extract the overview and add it to the list
            if 'overview' in movie_data and movie_data['overview']:
                overviews.append(movie_data['overview'])

            # Extract genres and add them to the genres list
            if 'genres' in movie_data:
                genres = [genre['name'] for genre in movie_data['genres']]
                genres_list.extend(genres)  # Add all genres for this movie to the list

            # Get actors from the credits endpoint
            credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?language=en-US"
            credits_response = requests.get(credits_url, headers=headers)

            if credits_response.status_code == 200:
                credits_data = credits_response.json()
                # Extract actor names from the cast list and add them to the actors list
                if 'cast' in credits_data:
                    actors = [actor['name'] for actor in credits_data['cast']]
                    actors_list.extend(actors)  # Add all actors for this movie to the list
        else:
            print(f"Failed to retrieve data for movie ID {movie_id}: {response.status_code}")

    # Count occurrences of each genre
    genre_counter = Counter(genres_list)
    genre_count_dict = dict(genre_counter)

    # Count occurrences of each actor
    actor_counter = Counter(actors_list)
    actor_count_dict = dict(actor_counter)

    # Print out the dictionary of genre counts
    print("\nGenre occurrences across all movies:")
    for genre, count in genre_count_dict.items():
        print(f"{genre}: {count}")

    # Print out the dictionary of actor counts
    print("\nActor occurrences across all movies:")
    for actor, count in actor_count_dict.items():
        print(f"{actor}: {count}")

    # Tokenize overviews and count word occurrences
    word_counter = Counter()

    for overview in overviews:
        doc = nlp(overview.lower)
        lemmas = [token.lemma_ for token in doc if token.is_alpha]
        word_counter.update(lemmas)

    # Convert the word counter to a dictionary
    word_count_dict = dict(word_counter)

    # Print out the dictionary of word counts
    print("\nWord occurrences across all movie overviews:")
    for word, count in word_count_dict.items():
        print(f"{word}: {count}")


if __name__ == '__main__':
    main()
