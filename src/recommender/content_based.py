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
            vectorizer = CountVectorizer(stop_words='english', max_features=2000)
            count_matrix = vectorizer.fit_transform(self.data['soup'])

            self.similarity_matrix = cosine_similarity(count_matrix, count_matrix)
            self.indices = pd.Series(self.data.index, index=self.data['title']).drop_duplicates()
        except Exception as e:
            raise CustomException(e, sys)

    def recommend(self, title, top_n=10):
        try:
            idx = self.indices.get(title)

            if idx is None or isinstance(idx, (pd.Series, list)) or hasattr(idx, "__len__") and len(idx) != 1:
                raise ValueError(f"Movie title '{title}' is ambiguous or not found in the dataset.")

            sim_scores = list(enumerate(self.similarity_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
            movie_indices = [i[0] for i in sim_scores]

            return self.data.iloc[movie_indices][['title', 'genres', 'overview']]
        except Exception as e:
            raise CustomException(e, sys)
