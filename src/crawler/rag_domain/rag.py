from llama_index.llms import Ollama
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import ServiceContext
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


llm = Ollama(model="mistral", request_timeout=30.0)
service_context = ServiceContext.from_defaults(llm="local")

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

query_engine = index.as_query_engine()
response = query_engine.query("Can you give me the summary of this Title IX Implemention of California?")
