from src.data.data_preprocessing import MovieDataPreprocessor
from src.recommender.content_based import ContentBasedRecommender

if __name__ == "__main__":
    processor = MovieDataPreprocessor(dataset_dir='dataset')
    
    # Load and process data
    movies_metadata, ratings, credits, keywords = processor.load_data()
    movies_metadata = processor.preprocess_movies(movies_metadata)
    merged_metadata = processor.merge_metadata(movies_metadata, credits, keywords)
    filtered_ratings = processor.filter_sparse_data(ratings)

    # Merge ratings with metadata
    final_data = processor.merge_with_ratings(filtered_ratings, merged_metadata)

    # Clean up missing values
    final_data = processor.handle_missing_values(final_data)

    # Generate soup column for content-based filtering
    final_data = processor.generate_soup(final_data)

    # Build and train recommender
    recommender = ContentBasedRecommender(final_data)
    recommender.train_model()

    # Recommend similar movies to a given title
    movie_title = "The Matrix"
    recommendations = recommender.recommend(movie_title, top_n=5)

    print(f"\nTop 5 movies similar to '{movie_title}':\n")
    print(recommendations)
