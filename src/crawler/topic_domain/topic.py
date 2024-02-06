import logging
import sys
import time

from llama_index import SimpleDirectoryReader
from llama_index.ingestion import IngestionPipeline
from llama_index.node_parser import SentenceSplitter
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
from bertopic import BERTopic

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

class TopicDistribution:
    def __init__(self):
        self.result = []
        start = time.time()
        transformations = [
            SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        ]
        documents = SimpleDirectoryReader("../output_domain").load_data()
        pipeline = IngestionPipeline(transformations=transformations)
        self.nodes = pipeline.run(documents=documents)
        end = time.time()
        print("Total Time taken - {} seconds".format(end - start))
        self.pre_df()

    def pre_df(self):
        states = []
        texts = []
        for node in self.nodes:
            file_path = node.metadata.get('file_path')
            state = file_path.split('/')[-1].split('.')[0]
            states.append(state)
            texts.append(node.text)
        data = {
            'states': states,
            'text': texts
        }
        self.df = pd.DataFrame.from_dict(data)
        print(self.df.head(5))

    def configure_embedding(self):
        embedding_model = SentenceTransformer("Cohere/Cohere-embed-english-v3.0")
        embeddings = embedding_model.encode(self.df['text'], show_progress_bar=True)
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=200, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        vectorizer_model = CountVectorizer(stop_words="english", min_df=5, ngram_range=(1, 2, 3, 4))
        keybert_model = KeyBERTInspired(top_n_words=30)
        pos_model = PartOfSpeech("en_core_web_sm")
        mmr_model = MaximalMarginalRelevance(diversity=0.5)
        combined_model = [KeyBERTInspired(top_n_words=30), MaximalMarginalRelevance(diversity=0.5)]
        representation_model = {
            "keyBERT": keybert_model,
            "mmr": mmr_model,
            "pos": pos_model,
            "combined": combined_model,
        }
        self.topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            top_n_words=10,
            verbose=True
        )
        topics, probs = self.topic_model.fit_transform(self.df['text'], embeddings)
        return topics, probs

topic = TopicDistribution()