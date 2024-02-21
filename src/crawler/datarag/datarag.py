import json
import logging
import sys
import time
import random
import requests
import chromadb
import pandas as pd
from llama_index import ServiceContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.ingestion import IngestionPipeline
from llama_index.llms import Ollama
from llama_index.node_parser import SemanticSplitterNodeParser
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import ChromaVectorStore

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

class Datarag:
    def __init__(self):
        start = time.time()
        llm = Ollama(model="mistral", request_timeout=1000)
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
        self.transformations = [
            SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model, num_workers=8),
        ]
        self.service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, transformations=self.transformations)
        self.reader_dict = {}
        self.result = []
        self.input_instances = self.get_instances()
        self.start()
        end = time.time()
        print("Total Time taken - {} seconds".format(end - start))

    def get_instances(self):
        response = requests.get('https://raw.githubusercontent.com/amir-karami/Workplace_Sexual_Harassment/master/EverySexsism-data-Workspace-Final.txt').content.decode('iso8859-1')
        questions = []
        data = response.split('\r\n')
        for i in range(0, 10):
            index = random.randrange(len(data))
            questions.append(data[index])
        return questions

    def start(self):
        df = pd.read_csv('../dataset_domain/data.csv')
        for state in ['utah']:
            print("Rag Stated for {}".format(state))
            documents = SimpleDirectoryReader(input_files=["../output_domain/{}.txt".format(state)]).load_data()
            documents = self.add_metadata(documents)
            self.reader_dict[state] = {'documents': documents}
            db = chromadb.PersistentClient(path="./{}".format(state))
            chroma_collection = db.get_or_create_collection("{}_title".format(state))
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            pipeline = IngestionPipeline(transformations=self.transformations, vector_store=vector_store)
            nodes = pipeline.run(documents=documents)    
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self.initialize_indexing(nodes, self.service_context, storage_context, state)

    def add_metadata(self, documents):
        for document in documents:
            file_path = document.metadata.get('file_path')
            state = file_path.split('/')[-1].split('.')[0]
            if state == 'federal':
                document.metadata['Topic'] = 'Title IX Implementation of {} State'.format(state)
                document.metadata['State'] = state
                document.metadata['Color'] = 'Neutral'
                document.metadata['Type'] = 'Neutral'
            elif state in ['california', 'new_york', 'maryland', 'massachusetts', 'washington']:
                document.metadata['Topic'] = 'Title IX Implementation of {} State'.format(state)
                document.metadata['State'] = state
                document.metadata['Color'] = 'Blue'
                document.metadata['Type'] = 'Democratic'
            else:
                document.metadata['Topic'] = 'Title IX Implementation of {} State'.format(state)
                document.metadata['State'] = state
                document.metadata['Color'] = 'Red'
                document.metadata['Type'] = 'Republican'
        return documents

    def initialize_indexing(self, nodes, service_context, storage_context, state):
        index = VectorStoreIndex(nodes=nodes, service_context=service_context, storage_context=storage_context)
        self.initialize_query_engine(index, state)

    def initialize_query_engine(self, index, state):
        self.query_engine = index.as_query_engine()
        self.reader_dict[state]['query_engine'] = self.query_engine
        self.pre_evaluation(state)

    def evaluate_instances(self, question):
        response = self.query_engine.query(question)
        self.result.append({
            'Question': question,
            'Response': response.response,
        })

    def pre_evaluation(self, state):
        for instance in self.input_instances:
            question = "Here is the example of people's experience of getting harassed: - \n{} \nCan you plan and provide the resolution of above harassment/discrimination based on Title IX Implementation in {}, considering that the same case happened in some univerity or in some workspace. Please provide the {} state specific resolution.".format(instance, state, state)
            self.evaluate_instances(question)
        with open("../datadump/{}.json".format(state), "w") as f:
            json.dump(self.result, f)
        with open("../datadump/{}.json".format(state), 'r') as openfile:
            json_object = json.load(openfile)
        final = ''
        for values in json_object:
            final = final + '\nQuestion : - {}\n\nAnswer : - {}\n'.format(values['Question'], values['Response'])
        with open("../datadump/{}.txt".format(state), "w", encoding="utf-8") as f:
            f.write(final)

    def evaluate(self, question):
        response = self.query_engine.query(question)
        print(response)

datarag = Datarag()
