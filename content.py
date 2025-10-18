from google import genai
from google.genai import types
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os


class SentenceClusterer:
    """
    Clusters semantically similar sentences using Google's GenAI embeddings.
    """

    def __init__(self, sentences, api_key=None,):
        self.sentences = sentences
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.client = genai.Client(api_key=self.api_key)
        

    def compute_embeddings(self):
        """
        Generate vector embeddings for the provided sentences.
        """
        embeddings = [
            np.array(e.values) for e in self.client.models.embed_content(
                model="gemini-embedding-001",
                contents=self.sentences,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
            ).embeddings
        ]
        return np.array(embeddings)

    def compute_similarity_matrix(self):
        """
        Compute cosine similarity matrix for all sentence embeddings.
        """
        embeddings = self.compute_embeddings()
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix

    def group_similar_sentences(self, threshold=0.7):
        """
        Group sentences that have cosine similarity >= threshold.
        """
        similarity_matrix = self.compute_similarity_matrix()
        groups = []

        for i, row in enumerate(similarity_matrix.tolist()):
            similar = [self.sentences[j] for j, score in enumerate(row) if score >= threshold]
            if similar not in groups:
                groups.append(similar)

        return groups

    def summarize_clusters(self,title=''):
        """
        Generate a short summary for each cluster using the Gemini model.
        """
        cluster_summary = {}

        for cluster in self.group_similar_sentences():
            summary = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents = f"""
You are analyzing responses to the survey question: "{title}"

Here are several responses: {cluster}

Write ONE short, clear summary sentence that captures the main shared idea among all these responses. 
Avoid repeating phrases and focus on the central theme.
"""

            )
            cluster_summary[summary.text] = len(cluster)

        return cluster_summary


