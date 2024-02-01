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

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

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
        documents = SimpleDirectoryReader("data/").load_data()
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
            if state in ['california']:
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
        with open("california_metrics.json", "w") as f:
            json.dump(self.result, f)

tam = TitleRagMetrics()
tam.extract_metadata()