from tmdbv3api import TMDb, Movie

tmdb = TMDb()
tmdb.api_key = "71ff4b710d0073a9c7a3868231901079"
tmdb.language = "en"
movie_api = Movie()

def get_movie_details(title):
    try:
        results = movie_api.search(title)
        if results:
            m = results[0]
            poster_url = f"https://image.tmdb.org/t/p/w500{m.poster_path}" if m.poster_path else None
            rating = m.vote_average
            release_year = m.release_date.split("-")[0] if m.release_date else "N/A"
            return {
                "poster_url": poster_url,
                "rating": rating,
                "release_year": release_year
            }
    except:
        pass
    return {
        "poster_url": None,
        "rating": "N/A",
        "release_year": "N/A"
    }

