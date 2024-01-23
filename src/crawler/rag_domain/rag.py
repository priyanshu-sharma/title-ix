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
    
    def pre_evaluation(self):
        response = self.query_engine.query("Can you give me the summary of this Title IX Implemention of California?")
        self.result.append(response)
        print(response)
        response = self.query_engine.query("Can you give me the summary of this Title IX Implemention of Texas?")
        self.result.append(response)
        print(response)
        response = self.query_engine.query("Can you give me the summary of this Title IX Implemention of Utah?")
        self.result.append(response)
        print(response)
        response = self.query_engine.query("How is the implementation of Title IX is different in California and Texas?")
        self.result.append(response)
        print(response)
        response = self.query_engine.query("How is the implementation of Title IX is different in Utah and Texas?")
        self.result.append(response)
        print(response)
        response = self.query_engine.query("How is the implementation of Title IX is different in California and Utah?")
        self.result.append(response)
        print(response)
        response = self.query_engine.query("How are Title IX Implementation is different in all three states, i.e. - California, Texas and Utah? List only the differences.")
        self.result.append(response)
        print(response)
        with open("output.json", "w") as f:
            json.dump(self.result, f)

    def evaluate(self, question):
        response = self.query_engine.query(question)
        print(response)

ta = TitleRag()
