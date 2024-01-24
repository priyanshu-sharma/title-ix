import json
import logging
import sys

from llama_index import ServiceContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.extractors import (EntityExtractor, KeywordExtractor,
                                    QuestionsAnsweredExtractor,
                                    SummaryExtractor, TitleExtractor)
from llama_index.ingestion import IngestionPipeline
from llama_index.llms import Ollama
from llama_index.node_parser import SemanticSplitterNodeParser

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

class TitleRag:
    def __init__(self):
        self.result = []
        self.cities = ['California', 'Texas', 'Utah', 'New York', 'Kansas', 'Maryland', 'Massachusetts', 'South Carolina', 'South Dakota', 'Washington']
        llm = Ollama(model="mistral", request_timeout=1000)
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
        transformations = [
            SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model),
            TitleExtractor(nodes=5, llm=llm),
            QuestionsAnsweredExtractor(questions=3, llm=llm),
            # EntityExtractor(prediction_threshold=0.5),
            # SummaryExtractor(summaries=["prev", "self", "next"], llm=llm),
            # KeywordExtractor(keywords=10, llm=llm),
        ]
        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, transformations=transformations)
        documents = SimpleDirectoryReader("../output_domain").load_data()
        documents = self.add_metadata(documents)
        print([document.metadata for document in documents])
        pipeline = IngestionPipeline(transformations=transformations)
        nodes = pipeline.run(documents=documents)
        print(nodes[0].metadata, nodes[2].metadata)
        # self.initialize_indexing(documents, service_context)

    def add_metadata(self, documents):
        for document in documents:
            file_path = document.metadata.get('file_path')
            state = file_path.split('/')[-1].split('.')[0]
            if state in ['california', 'new_york', 'maryland', 'massachusetts', 'washington']:
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

    def initialize_indexing(documents, service_context):
        index = VectorStoreIndex.from_documents(documents, service_context)
        self.initialize_query_engine(index)

    def initialize_query_engine(self, index):
        self.query_engine = index.as_query_engine()
        self.pre_evaluation()

    def type_one(self, city_one):
        question = 'Can you give me the summary of this Title IX Implemention of {}?'.format(city_one)
        response = self.query_engine.query(question)
        self.result.append({
            'Question': question,
            'Response': response.response,
        })

    def type_two(self, city_one, city_two):
        question = 'How is the implementation of Title IX is different in {} and {}?'.format(city_one, city_two)
        response = self.query_engine.query(question)
        self.result.append({
            'Question': question,
            'Response': response.response,
        })

    def type_three(self, total_cities, cities_string):
        question = 'How are Title IX Implementation is different in all {} states, i.e. - {}? List only the differences.'.format(total_cities, cities_string)
        response = self.query_engine.query(question)
        self.result.append({
            'Question': question,
            'Response': response.response,
        })
    
    def pre_evaluation(self):
        for city in self.cities:
            self.type_one(city)
        for i in range(0, len(self.cities)):
            for j in range(0, len(self.cities)):
                if self.cities[i] != self.cities[j]:
                    self.type_two(self.cities[i], self.cities[j])
        cities_string = ', '.join(str(city) for city in self.cities)
        self.type_three(len(self.cities), cities_string)
        with open("output.json", "w") as f:
            json.dump(self.result, f)

    def evaluate(self, question):
        response = self.query_engine.query(question)
        print(response)

ta = TitleRag()
