from llama_index.llms import Ollama
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import ServiceContext
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

class TitleRag:
    def __init__(self):
        llm = Ollama(model="mistral")
        service_context = ServiceContext.from_defaults(llm=llm, embed_model="local")
        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        self.query_engine = index.as_query_engine()
        self.pre_evaluation()
    
    def pre_evaluation(self):
        response = self.query_engine.query("Can you give me the summary of this Title IX Implemention of California?")
        print(response)
        response = self.query_engine.query("Can you give me the summary of this Title IX Implemention of Texas?")
        print(response)
        response = self.query_engine.query("How is the implementation of Title IX is different in California and Texas?")
        print(response)

    def evaluate(self, question):
        response = self.query_engine.query(question)
        print(response)

ta = TitleRag()

        