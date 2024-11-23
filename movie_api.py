import pandas as pd
import requests

from local.constants import TMDB_AUTHORIZATION
from movie_details import MovieDetails


class MovieAPI:
    def __init__(self, api_key):
        self.headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {api_key}"  # Replace with your API key
        }

    def fetch_movie_details(self, movie_id):
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US"
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to retrieve data for movie ID {movie_id}: {response.status_code}")
            return None

    def fetch_multiple_movies(self, tmdb_ids, movie_ids) -> pd.DataFrame:
        movie_details_list = []

        for tmdb_id, movie_id in zip(tmdb_ids, movie_ids):
            details = self.fetch_movie_details(tmdb_id)
            if details:
                movie_details = {
                    "movie_id": movie_id,
                    "tmdb_id": details["id"],
                    "title": details.get("title", ""),
                    "vote_average": details.get("vote_average", 0),
                    "vote_count": details.get("vote_count", 0),
                    "genere": details.get("genere")
                }
                movie_details_list.append(movie_details)

        movie_details_df = pd.DataFrame(movie_details_list,
                                        columns=["movie_id", "tmdb_id", "title", "vote_average", "vote_count", "genere"])
        return movie_details_df

