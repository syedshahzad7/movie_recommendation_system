from src.data.data_preprocessing import MovieDataPreprocessor

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

    print("Cleaned data preview:")
    print(final_data[['title', 'overview', 'release_date']].head())

