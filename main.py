from src.data.data_preprocessing import MovieDataPreprocessor
from src.recommender.content_based import ContentBasedRecommender

if __name__ == "__main__":
    processor = MovieDataPreprocessor(dataset_dir='dataset')

    # Step 1: Load raw data
    movies_metadata, ratings, credits, keywords = processor.load_data()

    # Step 2: Preprocess movie metadata
    movies_metadata = processor.preprocess_movies(movies_metadata)
    merged_metadata = processor.merge_metadata(movies_metadata, credits, keywords)

    # Step 3: Filter ratings (by active users and popular movies)
    filtered_ratings = processor.filter_sparse_data(ratings)

    # ✅ Limit rating size early to reduce downstream load
    filtered_ratings = filtered_ratings.sample(n=50000, random_state=42)

    # Step 4: Merge filtered ratings with metadata
    final_data = processor.merge_with_ratings(filtered_ratings, merged_metadata)

    # Step 5: Handle missing values
    final_data = processor.handle_missing_values(final_data)

    # Step 6: Generate 'soup' column for content-based filtering
    final_data = processor.generate_soup(final_data)

    # ✅ Sample small subset, drop duplicate titles, and reset index
    final_data = final_data.sample(n=3000, random_state=42).drop_duplicates(subset='title').reset_index(drop=True)

    # Step 7: Train recommender system
    recommender = ContentBasedRecommender(final_data)
    recommender.train_model()

    # Step 8: Get recommendations
    movie_title = "The Matrix"
    recommendations = recommender.recommend(movie_title, top_n=5)

    # Step 9: Display results
    print(f"\nTop 5 movies similar to '{movie_title}':\n")
    print(recommendations)
