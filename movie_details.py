class MovieDetails:
    def __init__(self, movie_data):
        # Id
        self.tmdb_id = movie_data.get("id", None)
        self.title = movie_data.get("title", None)

        # Numerical
        # self.adult = movie_data.get("adult", None)
        # self.budget = movie_data.get("budget", 0)
        # self.popularity = movie_data.get("popularity", 0)
        # self.release_date = movie_data.get("release_date", "")
        # self.runtime = movie_data.get("runtime", 0)
        self.vote_average = movie_data.get("vote_average", 0)
        self.vote_count = movie_data.get("vote_count", 0)

        # self.genres = set([genre['name'] for genre in movie_data.get("genres", [])])
        # self.origin_country = set([country['iso_3166_1'] for country in movie_data.get("production_countries", [])])
        # self.production_companies = set([company['name'] for company in movie_data.get("production_companies", [])])
        # self.spoken_languages = set([lang['name'] for lang in movie_data.get("spoken_languages", [])])

        # TF-IDF
        # self.overview = movie_data.get("overview", "")


    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "adult": self.adult,
            "budget": self.budget,
            "popularity": self.popularity,
            "release_date": self.release_date,
            "runtime": self.runtime,
            "vote_average": self.vote_average,
            "vote_count": self.vote_count
        }