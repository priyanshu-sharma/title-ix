import json
import logging
import sys
import time

import chromadb
from llama_index import ServiceContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.extractors import (EntityExtractor, KeywordExtractor,
                                    QuestionsAnsweredExtractor,
                                    SummaryExtractor, TitleExtractor)
from llama_index.ingestion import IngestionPipeline
from llama_index.llms import Ollama
from llama_index.node_parser import SemanticSplitterNodeParser
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import ChromaVectorStore

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

class TitleRag:
    def __init__(self):
        start = time.time()
        llm = Ollama(model="mistral", request_timeout=1000)
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
        transformations = [
            SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model, num_workers=8),
            TitleExtractor(nodes=5, llm=llm, num_workers=8),
            # QuestionsAnsweredExtractor(questions=3, llm=llm, num_workers=8),
            # EntityExtractor(prediction_threshold=0.5, num_workers=8),
            # SummaryExtractor(summaries=["prev", "self", "next"], llm=llm, num_workers=8),
            # KeywordExtractor(keywords=10, llm=llm, num_workers=8),
            # HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5"),
        ]
        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, transformations=transformations)
        documents = SimpleDirectoryReader("data/").load_data()
        print([document.metadata for document in documents])
        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_or_create_collection("testing")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        pipeline = IngestionPipeline(transformations=transformations, vector_store=vector_store)
        self.nodes = pipeline.run(documents=documents)
        print(self.nodes[0].metadata, self.nodes[2].metadata)
    #     storage_context = StorageContext.from_defaults(vector_store=vector_store)
    #     self.initialize_indexing(nodes, service_context, storage_context)
        end = time.time()
        print("Total Time taken - {} seconds".format(end - start))

    # def initialize_indexing(self, nodes, service_context, storage_context):
    #     index = VectorStoreIndex(nodes=nodes, service_context=service_context, storage_context=storage_context)
    #     self.initialize_query_engine(index)

    # def initialize_query_engine(self, index):
    #     self.query_engine = index.as_query_engine()

ta = TitleRag()
