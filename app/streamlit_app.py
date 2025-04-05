import streamlit as st
from src.data.data_preprocessing import MovieDataPreprocessor
from src.recommender.content_based import ContentBasedRecommender
from src.utils.tmdb_utils import get_movie_details
import pandas as pd

# Set Streamlit page configuration
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender System")
st.markdown("Select a movie you like and we'll suggest similar ones!")

# Cache model loading, but suppress default cache message
@st.cache_resource(show_spinner=False)
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

# Show friendly loading spinner
with st.spinner("🚀 Loading recommendation model..."):
    recommender, final_data = load_model()

# Movie selection UI
movie_list = final_data['title'].sort_values().tolist()
selected_movie = st.selectbox("🎥 Choose a movie:", movie_list)

# Display recommendations
if st.button("Get Recommendations"):
    try:
        recommendations = recommender.recommend(selected_movie, top_n=8)
        

        # 2x2 layout: display rows of 2 movie cards
        rows = [recommendations.iloc[i:i+2] for i in range(0, len(recommendations), 2)]
        for row_df in rows:
            cols = st.columns(2)
            for col, (_, row) in zip(cols, row_df.iterrows()):
                with col:
                    details = get_movie_details(row['title'])
                    poster_col, info_col = st.columns([1, 3])
                    
                    with poster_col:
                        if details['poster_url']:
                            st.image(details['poster_url'], width=160)
                        else:
                            st.markdown("🚫 No poster available")

                    with info_col:
                        st.markdown(f"### {row['title']}")
                        st.markdown(f"**🎬 Genre:** {', '.join(row['genres'])}")
                        st.markdown(f"**📅 Year:** {details['release_year']}")
                        st.markdown(f"**⭐ IMDb Rating:** {details['rating']}")
                        st.markdown(
                            f"<div style='font-size: 0.85rem;' title='{row['overview']}'>"
                            f"{row['overview'][:200]}..."
                            f"</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
