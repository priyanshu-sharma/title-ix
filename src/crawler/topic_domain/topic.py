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
from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline
from bertopic.representation import TextGeneration


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

class TopicDistribution:
    def __init__(self):
        self.result = []
        start = time.time()
        transformations = [
            SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        ]
        documents = SimpleDirectoryReader("../output_domain/federal/").load_data()
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
        hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 5))
        model = AutoModelForCausalLM.from_pretrained(
            "TheBloke/zephyr-7B-alpha-GGUF",
            model_file="zephyr-7b-alpha.Q4_K_M.gguf",
            model_type="mistral",
            gpu_layers=50,
            hf=True,
            context_length = 6000,
        )
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")
        generator = pipeline(
            model=model, tokenizer=tokenizer,
            task='text-generation',
            max_new_tokens=50,
            repetition_penalty=1.1
        )
        system_prompt = """
        <s>[INST] <<SYS>>
        You are a helpful, respectful and honest assistant for labeling topics.
        <</SYS>>
        """
        example_prompt = """
        I have a topic that contains the following documents:
        - Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
        - Meat, but especially beef, is the word food in terms of emissions.
        - Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

        The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

        Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.

        [/INST] Environmental impacts of eating meat
        """
        main_prompt = """
        [INST]
        I have a topic that contains the following documents:
        [DOCUMENTS]

        The topic is described by the following keywords: '[KEYWORDS]'.

        Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.
        [/INST]
        """
        prompt = system_prompt + example_prompt + main_prompt
        zephyr = TextGeneration(generator, prompt=prompt)
        keybert_model = KeyBERTInspired(top_n_words=30)
        mmr_model = MaximalMarginalRelevance(diversity=0.3)
        combined_model = [KeyBERTInspired(top_n_words=30), MaximalMarginalRelevance(diversity=0.3)]
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
            # top_n_words=10,
            verbose=True
        )
        topics, probs = self.topic_model.fit_transform(self.texts, embeddings)
        return topics, probs

topic = TopicDistribution()
topics, probs = topic.configure_embedding()
df = topic.topic_model.get_topic_info()
print(df.columns)
df.to_csv('topics.csv')
data = {'texts': topic.texts, 'states': topic.states, 'topics': topics, 'probability': probs}
df = pd.DataFrame.from_dict(data)
df.to_csv('data.csv')
topic.topic_model.save("data", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)
# Visualize topic per class