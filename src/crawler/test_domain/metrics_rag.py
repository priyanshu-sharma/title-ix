import json
import logging
import sys
import time

import chromadb
from llama_index import ServiceContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.ingestion import IngestionPipeline
from llama_index.llms import Ollama
from llama_index.node_parser import SemanticSplitterNodeParser
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import ChromaVectorStore
from custom_transformation import TextBlobTransformation, VaderTransformation, RobertaTranformation, BertTransformation
import pandas as pd
import plotly.graph_objects as go
import numpy as np

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def calculate_average(city_data, key):
    city_key = city_data[key]
    key_total = 0
    for data in city_key:
        key_total = key_total + float(data)
    return key_total/len(city_key)

def metrics_to_data():
    with open('metrics.json', 'r') as openfile:
        json_object = json.load(openfile)
    overall_result = {}
    for city, city_data in json_object.items():
        overall_result[city] = []
        keys = ['textblob_polarity', 'textblob_subjectivity', 'vader_negative', 'vader_positive', 'vader_neutral', 'vader_compound', 'roberta_negative', 'roberta_neutral', 'roberta_positive', 'bert_left', 'bert_center', 'bert_right']
        for key in keys:
            overall_result[city].append(calculate_average(city_data, key))
        overall_result['feature'] = keys
    df = pd.DataFrame.from_dict(overall_result)
    print(df.head(10), df.columns)
    return df, overall_result

def red_blue(df):
    cf = df
    cf['red'] = 0
    cf['blue'] = 0
    cf['blue'] = cf['california'] + cf['maryland'] + cf['massachusetts'] + cf['new_york'] + cf['washington']
    cf['red'] = cf['kansas'] + cf['south_carolina'] + cf['south_dakota'] + cf['texas'] + cf['utah']
    cf['red'] = cf['red']/5
    cf['blue'] = cf['blue']/5
    cf = cf.drop(['california', 'kansas', 'maryland', 'massachusetts', 'new_york', 'south_carolina', 'south_dakota', 'texas', 'utah', 'washington'], axis=1)
    categories = cf['feature']
    fig = go.Figure()
    columns = ['blue', 'red']
    for column in columns:
        fig.add_trace(go.Scatterpolar(r=cf[column], theta=categories, name=column))        
    fig.show()
    fig.write_image("red_blue_metrics.png")

def radar_chat(df):
    categories = df['feature']
    fig = go.Figure()
    for column in df.columns:
        fig.add_trace(go.Scatterpolar(r=df[column], theta=categories, name=column))        
    fig.show()
    fig.write_image("metrics.png")

def sortdata(overall_result, ikey):
    r = {}
    keys = ['textblob_polarity', 'textblob_subjectivity', 'vader_negative', 'vader_positive', 'vader_neutral', 'vader_compound', 'roberta_negative', 'roberta_neutral', 'roberta_positive', 'bert_left', 'bert_center', 'bert_right']
    i = 0
    for key in keys:
        if key != ikey:
            i = i + 1
        else:
            break
    print(i, keys[i])
    for k, v in overall_result.items():
        if k != 'feature':
            r[k] = np.float64(v[i])
    return dict(sorted(r.items(), key=lambda item: item[1]))

class TitleRagMetrics:
    def __init__(self):
        self.result = {}
        start = time.time()
        llm = Ollama(model="mistral", request_timeout=1000)
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
        transformations = [
            SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model, num_workers=8),
            TextBlobTransformation(),
            VaderTransformation(),
            RobertaTranformation(),
            BertTransformation(),
        ]
        documents = SimpleDirectoryReader("../output_domain").load_data()
        # print([document.metadata for document in documents])
        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_or_create_collection("title_ix")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        pipeline = IngestionPipeline(transformations=transformations, vector_store=vector_store)
        self.nodes = pipeline.run(documents=documents)
        # print(self.nodes[0].metadata.keys())
        end = time.time()
        print("Total Time taken - {} seconds".format(end - start))
        # print(len(self.nodes))
        # for node in self.nodes:
        #     print(node.metadata['bert'])

    def extract_metadata(self):
        for node in self.nodes:
            file_path = node.metadata.get('file_path')
            state = file_path.split('/')[-1].split('.')[0]
            if state not in self.result.keys():
                self.result[state] = {
                    'textblob_polarity': [str(node.metadata['textblob'].get('polarity'))],
                    'textblob_subjectivity': [str(node.metadata['textblob'].get('subjectivity'))],
                    'vader_negative': [str(node.metadata['vader'].get('negative'))],
                    'vader_positive': [str(node.metadata['vader'].get('positive'))],
                    'vader_neutral': [str(node.metadata['vader'].get('neutral'))],
                    'vader_compound': [str(node.metadata['vader'].get('compound'))],
                    'roberta_negative': [str(node.metadata['roberta'].get('negative'))],
                    'roberta_neutral': [str(node.metadata['roberta'].get('neutral'))],
                    'roberta_positive': [str(node.metadata['roberta'].get('positive'))],
                    'bert_left': [str(node.metadata['bert'].get('left'))],
                    'bert_center': [str(node.metadata['bert'].get('center'))],
                    'bert_right': [str(node.metadata['bert'].get('right'))],
                }
            else:
                self.result[state]['textblob_polarity'].append(str(node.metadata['textblob'].get('polarity')))
                self.result[state]['textblob_subjectivity'].append(str(node.metadata['textblob'].get('subjectivity')))
                self.result[state]['vader_negative'].append(str(node.metadata['vader'].get('negative')))
                self.result[state]['vader_positive'].append(str(node.metadata['vader'].get('positive')))
                self.result[state]['vader_neutral'].append(str(node.metadata['vader'].get('neutral')))
                self.result[state]['vader_compound'].append(str(node.metadata['vader'].get('compound')))
                self.result[state]['roberta_negative'].append(str(node.metadata['roberta'].get('negative')))
                self.result[state]['roberta_neutral'].append(str(node.metadata['roberta'].get('neutral')))
                self.result[state]['roberta_positive'].append(str(node.metadata['roberta'].get('positive')))
                self.result[state]['bert_left'].append(str(node.metadata['bert'].get('left')))
                self.result[state]['bert_center'].append(str(node.metadata['bert'].get('center')))
                self.result[state]['bert_right'].append(str(node.metadata['bert'].get('right')))
        with open("metrics.json", "w") as f:
            json.dump(self.result, f)

# tam = TitleRagMetrics()
# tam.extract_metadata()