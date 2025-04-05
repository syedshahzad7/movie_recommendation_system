import streamlit as st
from src.data.data_preprocessing import MovieDataPreprocessor
from src.recommender.content_based import ContentBasedRecommender
from src.utils.tmdb_utils import get_movie_details
import pandas as pd
from string import Template

# Set Streamlit page configuration
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.markdown("""
    <h1 style='font-size: 40px;'>🎬 Movie Recommender System</h1>
    <p style='font-size: 16px;'>Select a movie you like and we'll suggest similar ones!</p>
""", unsafe_allow_html=True)

# Load styles from external CSS
with open("app/assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Cache model loading, but suppress default cache message
@st.cache_resource(show_spinner=False)
def load_model():
    import joblib

    final_data = pd.read_pickle("saved_models/final_data.pkl")
    similarity_matrix = joblib.load("saved_models/similarity_matrix.pkl")
    indices = joblib.load("saved_models/indices.pkl")

    recommender = ContentBasedRecommender(final_data)
    recommender.similarity_matrix = similarity_matrix
    recommender.indices = indices

    return recommender, final_data


# Show friendly loading spinner
with st.spinner("🚀 Loading recommendation model..."):
    recommender, final_data = load_model()

# Load HTML template
with open("app/assets/movie_card.html", "r", encoding="utf-8") as file:
    html_template = Template(file.read())

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
                    poster_url = details['poster_url'] or "https://via.placeholder.com/150x220?text=No+Image"
                    genres = ', '.join(row['genres']) if isinstance(row['genres'], list) else row['genres']

                    # Fill HTML template
                    html = html_template.substitute(
                        poster_url=poster_url,
                        title=row['title'],
                        rating=details['rating'],
                        year=details['release_year'],
                        genres=genres,
                        overview=row['overview'][:250] + '...'
                    )
                    st.markdown(html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
