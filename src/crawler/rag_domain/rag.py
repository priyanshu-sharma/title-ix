from llama_index.llms import Ollama
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import ServiceContext
import logging
import sys
import json

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

class TitleRag:
    def __init__(self):
        llm = Ollama(model="mistral")
        self.result = []
        service_context = ServiceContext.from_defaults(llm=llm, embed_model="local")
        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        self.query_engine = index.as_query_engine()
        self.pre_evaluation()

    def type_one(self, city_one):
        question = 'Can you give me the summary of this Title IX Implemention of {}?'.format(city_one)
        response = self.query_engine.query(question).json()
        self.result.append({
            'Question': question,
            'Response': response
        })

    def type_two(self, city_one, city_two):
        question = 'How is the implementation of Title IX is different in {} and {}?'.format(city_one, city_two)
        response = self.query_engine.query(question)
        self.result.append({
            'Question': question,
            'Response': response
        })

    def type_three(self, total, cities):
        question = 'How are Title IX Implementation is different in all {} states, i.e. - {}? List only the differences.'.format(total, cities)
        response = self.query_engine.query(question)
        self.result.append({
            'Question': question,
            'Response': response
        })
    
    def pre_evaluation(self):
        self.type_one('California')
        # self.type_one('Texas')
        # self.type_one('Utah')
        # self.type_two('California', 'Texas')
        # self.type_two('Utah', 'Texas')
        # self.type_two('Utah', 'California')
        # self.type_three('three', 'California, Texas and Utah')
        with open("output.json", "w") as f:
            json.dump(self.result, f)

    def evaluate(self, question):
        response = self.query_engine.query(question)
        print(response)

ta = TitleRag()
