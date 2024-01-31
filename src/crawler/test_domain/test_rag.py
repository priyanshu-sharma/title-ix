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
from custom_transformation import TextBlobTransformation, VaderTransformation, RobertaTranformation, BertTransformation

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

class TitleRagQA:
    def __init__(self):
        self.result = {}
        start = time.time()
        llm = Ollama(model="mistral", request_timeout=1000)
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
        transformations = [
            SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model, num_workers=8),
            # TitleExtractor(nodes=5, llm=llm, num_workers=8),
            # QuestionsAnsweredExtractor(questions=5, llm=llm, num_workers=8),
            # EntityExtractor(prediction_threshold=0.5, num_workers=8),
            # SummaryExtractor(summaries=["prev", "self", "next"], llm=llm, num_workers=8),
            # KeywordExtractor(keywords=10, llm=llm, num_workers=8),
            TextBlobTransformation(),
            VaderTransformation(),
            RobertaTranformation(),
            BertTransformation(),
        ]
        # service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, transformations=transformations)
        documents = SimpleDirectoryReader("data/").load_data()
        print([document.metadata for document in documents])
        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_or_create_collection("testing")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        pipeline = IngestionPipeline(transformations=transformations, vector_store=vector_store)
        self.nodes = pipeline.run(documents=documents)
        print(self.nodes[0].metadata.keys())
        end = time.time()
        print("Total Time taken - {} seconds".format(end - start))
        print(len(self.nodes))
        for node in self.nodes:
            print(node.metadata['bert'])

    def extract_metadata(self):
        for node in self.nodes:
            file_path = node.metadata.get('file_path')
            state = file_path.split('/')[-1].split('.')[0]
            if state in ['california']:
                if state not in self.result.keys():
                    self.result[state] = {
                        # 'document_titles': {node.metadata.get('document_title')},
                        # 'question_answers': {node.metadata.get('questions_this_excerpt_can_answer')},
                        # 'entities': [entity for entity in node.metadata.get('entities')],
                        # 'next_section_summary': {node.metadata.get('next_section_summary')},
                        # 'section_summary': {node.metadata.get('section_summary')},
                        # 'keywords': {node.metadata.get('excerpt_keywords')},
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
                    # self.result[state]['entities'] = set(self.result[state]['entities'])
                else:
                    # self.result[state]['document_titles'].add(node.metadata.get('document_title'))
                    # self.result[state]['question_answers'].add(node.metadata.get('questions_this_excerpt_can_answer'))
                    # for entity in node.metadata.get('entities'):
                    #     self.result[state]['entities'].add(entity)
                    # self.result[state]['next_section_summary'].add(node.metadata.get('next_section_summary'))
                    # self.result[state]['section_summary'].add(node.metadata.get('section_summary'))
                    # self.result[state]['keywords'].add(node.metadata.get('excerpt_keywords'))
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
        # self.result[state]['document_titles'] = list(self.result[state]['document_titles'])
        # self.result[state]['question_answers'] = list(self.result[state]['question_answers'])
        # self.result[state]['entities'] = list(self.result[state]['entities'])
        # self.result[state]['next_section_summary'] = list(self.result[state]['next_section_summary'])
        # self.result[state]['section_summary'] = list(self.result[state]['section_summary'])
        # self.result[state]['keywords'] = list(self.result[state]['keywords'])
        with open("california_qa.json", "w") as f:
            json.dump(self.result, f)

ta = TitleRagQA()
ta.extract_metadata()