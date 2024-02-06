import logging
import sys
import time

import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from hdbscan import HDBSCAN
from llama_index import SimpleDirectoryReader
from llama_index.ingestion import IngestionPipeline
from llama_index.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from bertopic.representation import TextGeneration
from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline


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
        self.states = []
        self.texts = []
        for node in self.nodes:
            file_path = node.metadata.get('file_path')
            state = file_path.split('/')[-1].split('.')[0]
            self.states.append(state)
            self.texts.append(node.text)

    def configure_embedding(self):
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedding_model.encode(self.texts, show_progress_bar=True)
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=150, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 5))
        keybert_model = KeyBERTInspired(top_n_words=30)
        mmr_model = MaximalMarginalRelevance(diversity=0.3)
        combined_model = [KeyBERTInspired(top_n_words=30), MaximalMarginalRelevance(diversity=0.3)]
        model = AutoModelForCausalLM.from_pretrained(
            "TheBloke/zephyr-7B-alpha-GGUF",
            model_file="zephyr-7b-alpha.Q4_K_M.gguf",
            model_type="mistral",
            gpu_layers=50,
            hf=True
        )
        model.max_seq_len = 4096
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")
        generator = pipeline(
            model=model, tokenizer=tokenizer,
            task='text-generation',
            max_new_tokens=50,
            repetition_penalty=1.1
        )
        prompt = """<|system|>You are a helpful, respectful and honest assistant for labeling topics..</s>
        <|user|>
        I have a topic that contains the following documents:
        [DOCUMENTS]

        The topic is described by the following keywords: '[KEYWORDS]'.

        Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.</s>
        <|assistant|>"""

        zephyr = TextGeneration(generator, prompt=prompt)
        representation_model = {
            "keyBERT": keybert_model,
            "mmr": mmr_model,
            "combined": combined_model,
            "zephyr": zephyr,
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
        topics, probs = self.topic_model.fit_transform(self.texts, embeddings)
        return topics, probs

topic = TopicDistribution()