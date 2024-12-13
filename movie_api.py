import pandas as pd
import requests
import os
import json

from local.constants import TMDB_AUTHORIZATION


class MovieAPI:
    def __init__(self, cache_file="movie_cache.json"):
        self.headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {TMDB_AUTHORIZATION}"  # Replace with your API key
        }
        self.cache_file = cache_file
        self.movie_cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.movie_cache, f, indent=4)

    def fetch_movie(self, movie_id):
        if str(movie_id) in self.movie_cache:
            print(f"Loading movie ID {movie_id} from cache.")
            return self.movie_cache[str(movie_id)]

        url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US"
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            movie_details = response.json()
            self.movie_cache[str(movie_id)] = movie_details  # Save to cache
            self._save_cache()  # Persist cache
            return movie_details
        else:
            print(f"Failed to retrieve data for movie ID {movie_id}: {response.status_code}")
            return None

    def fetch_movies(self, movie_csv) -> pd.DataFrame:
        movie_details_list = []
        for _, movie in movie_csv.iterrows():
            id = movie["id"]
            tmdb_id = movie["tmdb_id"]
            details = self.fetch_movie(tmdb_id)
            if details:
                movie_details = {
                    "id": id,
                    "tmdb_id": tmdb_id,
                    "title": details.get("title"),
                    "vote_average": details.get("vote_average"),
                    "vote_count": details.get("vote_count"),
                    "genres": [genre["name"] for genre in details.get("genres", [])],
                }
                movie_details_list.append(movie_details)

        movies_df = pd.DataFrame(
            movie_details_list,
            columns=["id", "tmdb_id", "title", "vote_average", "vote_count", "genres"],
        ).set_index("id")
        return movies_df
