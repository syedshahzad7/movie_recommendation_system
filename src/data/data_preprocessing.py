import pandas as pd
import numpy as np
import os
import ast
from src.logger import logging
from src.exception import CustomException
import sys

class MovieDataPreprocessor:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
    
    def load_data(self):
        try:
            movies_metadata = pd.read_csv(os.path.join(self.dataset_dir, 'movies_metadata.csv'), low_memory=False)
            ratings = pd.read_csv(os.path.join(self.dataset_dir, 'ratings.csv'))
            credits = pd.read_csv(os.path.join(self.dataset_dir, 'credits.csv'))
            keywords = pd.read_csv(os.path.join(self.dataset_dir, 'keywords.csv'))
            logging.info("All dataset files loaded successfully.")
            return movies_metadata, ratings, credits, keywords
        except Exception as e:
            raise CustomException(e, sys)
    
    def preprocess_movies(self, movies_metadata):
        try:
            # Fix errors in numeric columns
            movies_metadata['id'] = pd.to_numeric(movies_metadata['id'], errors='coerce')
            movies_metadata = movies_metadata.dropna(subset=['id'])
            movies_metadata['id'] = movies_metadata['id'].astype(int)

            # Convert genres from string to list
            movies_metadata['genres'] = movies_metadata['genres'].apply(
                lambda x: [i['name'] for i in ast.literal_eval(x)] if pd.notnull(x) else []
            )
            logging.info("Preprocessing of movies_metadata completed.")
            return movies_metadata
        except Exception as e:
            raise CustomException(e, sys)

    def merge_metadata(self, movies, credits, keywords):
        try:
            credits['id'] = credits['id'].astype(int)
            keywords['id'] = keywords['id'].astype(int)

            merged = movies.merge(credits, on='id')
            merged = merged.merge(keywords, on='id')

            # Parse cast and crew JSONs
            merged['cast'] = merged['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)] if pd.notnull(x) else [])
            merged['crew'] = merged['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)] if pd.notnull(x) else [])
            merged['keywords'] = merged['keywords'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)] if pd.notnull(x) else [])

            logging.info("Metadata, credits, and keywords successfully merged.")
            return merged
        except Exception as e:
            raise CustomException(e, sys)
        
    def filter_sparse_data(self, ratings, min_user_ratings=20, min_movie_ratings=50):
        try:
            # Filter out users with too few ratings
            user_counts = ratings['userId'].value_counts()
            active_users = user_counts[user_counts >= min_user_ratings].index
            ratings = ratings[ratings['userId'].isin(active_users)]

            # Filter out movies with too few ratings
            movie_counts = ratings['movieId'].value_counts()
            popular_movies = movie_counts[movie_counts >= min_movie_ratings].index
            ratings = ratings[ratings['movieId'].isin(popular_movies)]

            logging.info(f"Filtered ratings to {ratings.shape[0]} entries with active users and popular movies.")
            return ratings
        except Exception as e:
            raise CustomException(e, sys)

        
    def merge_with_ratings(self, filtered_ratings, merged_metadata):
        try:
            # 'id' in merged_metadata is from movies_metadata (TMDb)
            # 'movieId' in ratings is from MovieLens
            # We need to use the links.csv file to connect them

            links_df = pd.read_csv(os.path.join(self.dataset_dir, 'links.csv'))
            links_df = links_df[['movieId', 'tmdbId']].dropna()
            links_df['tmdbId'] = links_df['tmdbId'].astype(int)

            # Rename for merge compatibility
            links_df.rename(columns={'tmdbId': 'id'}, inplace=True)

            # Merge to get 'id' in ratings
            ratings_with_tmdb = filtered_ratings.merge(links_df, on='movieId', how='inner')
            logging.info(f"Ratings successfully mapped to TMDB IDs. Shape: {ratings_with_tmdb.shape}")

            # Final merge with movie metadata
            # Select only relevant columns from metadata
            columns_to_keep = ['id', 'title', 'genres', 'overview', 'cast', 'crew', 'keywords', 'release_date', 'runtime']
            merged_metadata_small = merged_metadata[columns_to_keep]

            # Merge only on selected columns
            final_df = ratings_with_tmdb.merge(merged_metadata_small, on='id', how='inner')

            logging.info(f"Final merged dataset shape: {final_df.shape}")
            
            return final_df
        except Exception as e:
            raise CustomException(e, sys)
        
    def handle_missing_values(self, df):
        try:
            # Fill missing overviews
            df['overview'] = df['overview'].fillna("No overview available")

            # Parse release_date and drop rows where it’s still missing
            df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
            df = df[df['release_date'].notnull()]

            logging.info("Missing values handled: overview filled, invalid release dates removed.")
            return df
        except Exception as e:
            raise CustomException(e, sys)
        
    def generate_soup(self, df):
        try:
            # Clean list-based columns without apply
            def clean_list_column_manual(col):
                cleaned = []
                for row in col:
                    if isinstance(row, list):
                        cleaned.append([i.replace(" ", "").lower() for i in row])
                    else:
                        cleaned.append([])
                return cleaned

            df['genres'] = clean_list_column_manual(df['genres'])
            df['keywords'] = clean_list_column_manual(df['keywords'])

            # Keep top 3 cast and top 1 crew
            df['cast'] = clean_list_column_manual(df['cast'])
            df['cast'] = [cast[:3] for cast in df['cast']]

            df['crew'] = clean_list_column_manual(df['crew'])
            df['crew'] = [crew[:1] for crew in df['crew']]

            # Build soup manually
            soup_list = []
            for i in range(len(df)):
                genres = ' '.join(df.loc[i, 'genres'])
                keywords = ' '.join(df.loc[i, 'keywords'])
                cast = ' '.join(df.loc[i, 'cast'])
                crew = ' '.join(df.loc[i, 'crew'])
                overview = df.loc[i, 'overview'].lower()

                soup = f"{genres} {keywords} {cast} {crew} {overview}"
                soup_list.append(soup)

            df['soup'] = soup_list

            return df
        except Exception as e:
            raise CustomException(e, sys)



