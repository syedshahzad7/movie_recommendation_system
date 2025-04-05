import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.exception import CustomException
import sys

class ContentBasedRecommender:
    def __init__(self, data):
        self.data = data
        self.similarity_matrix = None
        self.indices = None

    def train_model(self):
        try:
            # Create vectorizer
            vectorizer = CountVectorizer(stop_words='english')
            count_matrix = vectorizer.fit_transform(self.data['soup'])

            # Compute cosine similarity
            self.similarity_matrix = cosine_similarity(count_matrix, count_matrix)

            # Create reverse index for title lookup
            self.indices = pd.Series(self.data.index, index=self.data['title']).drop_duplicates()
        except Exception as e:
            raise CustomException(e, sys)

    def recommend(self, title, top_n=10):
        try:
            idx = self.indices.get(title)
            if idx is None:
                return []

            sim_scores = list(enumerate(self.similarity_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
            movie_indices = [i[0] for i in sim_scores]

            return self.data.iloc[movie_indices][['title', 'genres', 'overview']]
        except Exception as e:
            raise CustomException(e, sys)
