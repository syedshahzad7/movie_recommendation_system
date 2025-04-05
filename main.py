from src.data.data_preprocessing import MovieDataPreprocessor
from src.recommender.content_based import ContentBasedRecommender
import pandas as pd
import joblib
import os

if __name__ == "__main__":
    processor = MovieDataPreprocessor(dataset_dir='dataset')

    # Step 1: Load raw data
    movies_metadata, ratings, credits, keywords = processor.load_data()

    # Step 2: Preprocess movie metadata
    movies_metadata = processor.preprocess_movies(movies_metadata)
    merged_metadata = processor.merge_metadata(movies_metadata, credits, keywords)

    # Step 3: Filter ratings (by active users and popular movies)
    filtered_ratings = processor.filter_sparse_data(ratings)
    filtered_ratings = filtered_ratings.sample(n=50000, random_state=42)

    # Step 4: Merge filtered ratings with metadata
    final_data = processor.merge_with_ratings(filtered_ratings, merged_metadata)

    # Step 5: Handle missing values
    final_data = processor.handle_missing_values(final_data)

    # Step 6: Generate 'soup' column for content-based filtering
    final_data = processor.generate_soup(final_data)

    # ✅ Sample and cleanup
    final_data = final_data.sample(n=15000, random_state=42).drop_duplicates(subset='title').reset_index(drop=True)

    # Step 7: Train recommender
    recommender = ContentBasedRecommender(final_data)
    recommender.train_model()

    # ✅ Step 8: Save model and data
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(recommender.similarity_matrix, "saved_models/similarity_matrix.pkl")
    joblib.dump(recommender.indices, "saved_models/indices.pkl")
    final_data.to_pickle("saved_models/final_data.pkl")

    print("✅ Model and data saved successfully!")
