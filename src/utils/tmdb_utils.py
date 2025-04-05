from tmdbv3api import TMDb, Movie
import requests

tmdb = TMDb()
tmdb.api_key = "71ff4b710d0073a9c7a3868231901079"  
tmdb.language = "en"
movie_api = Movie()

def get_poster_url(title):
    try:
        results = movie_api.search(title)
        if results:
            poster_path = results[0].poster_path
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        pass
    return None
