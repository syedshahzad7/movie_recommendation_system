import streamlit as st
from src.data.data_preprocessing import MovieDataPreprocessor
from src.recommender.content_based import ContentBasedRecommender
from src.utils.tmdb_utils import get_poster_url  # 👈 TMDb poster fetch
import pandas as pd

# Page config
st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommender System")
st.markdown("Select a movie you like and we'll suggest similar ones!")

@st.cache_resource
def load_model():
    processor = MovieDataPreprocessor(dataset_dir='dataset')
    movies_metadata, ratings, credits, keywords = processor.load_data()
    movies_metadata = processor.preprocess_movies(movies_metadata)
    merged_metadata = processor.merge_metadata(movies_metadata, credits, keywords)
    filtered_ratings = processor.filter_sparse_data(ratings)
    filtered_ratings = filtered_ratings.sample(n=50000, random_state=42)
    final_data = processor.merge_with_ratings(filtered_ratings, merged_metadata)
    final_data = processor.handle_missing_values(final_data)
    final_data = processor.generate_soup(final_data)
    final_data = final_data.sample(n=15000, random_state=42).drop_duplicates(subset='title').reset_index(drop=True)

    recommender = ContentBasedRecommender(final_data)
    recommender.train_model()
    return recommender, final_data

recommender, final_data = load_model()

movie_list = final_data['title'].sort_values().tolist()
selected_movie = st.selectbox("🎥 Choose a movie:", movie_list)

if st.button("Get Recommendations"):
    try:
        recommendations = recommender.recommend(selected_movie, top_n=5)
        st.success(f"Top 5 movies similar to **{selected_movie}**:")

        for i, row in recommendations.iterrows():
            col1, col2 = st.columns([1, 4])
            with col1:
                poster_url = get_poster_url(row['title'])
                if poster_url:
                    st.image(poster_url, use_container_width=True)
                else:
                    st.text("No poster available")

            with col2:
                st.markdown(f"**🎞️ {row['title']}**")
                st.markdown(f"*Genres:* {', '.join(row['genres'])}")
                st.markdown(row['overview'])
            st.markdown("---")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
