import autogen
from autogen.agentchat.contrib.retrieve_assistant_agent import \
    RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import \
    RetrieveUserProxyAgent

import chromadb


class TitleIXRag:
    def __init__(self):
        self.config_list = [{"base_url": "http://0.0.0.0:8000", "model": "mistral"}]
        self.llm_config = {
            "cache_seed": 42,  
            "temperature": 0,
            "config_list": self.config_list,
            "timeout": 120,
        }

        self.texas_assistant = RetrieveAssistantAgent(
            name="assistant",
            system_message="You are a helpful assistant.",
            llm_config=self.llm_config,
        )

        self.texas_ragproxyagent = RetrieveUserProxyAgent(
            name="ragproxyagent",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            retrieve_config={
                "task": "qa",
                "docs_path": "https://raw.githubusercontent.com/priyanshu-sharma/title-ix/master/src/crawler/dataset_domain/output/california.txt",
                "must_break_at_empty_line": True,
                "model": "mistral",
                "client": chromadb.PersistentClient(path="./db"),
                "collection_name": "california-titleix",
                "embedding_model": "all-MiniLM-L6-v2"
            },
        )

    def chat(self, problem):
        self.texas_assistant.reset()
        self.texas_ragproxyagent.initiate_chat(self.texas_assistant, problem=problem)

title = TitleIXRag()
title.chat("List all the importrant")
title.chat("Give me a brief summary of Title IX implementation in California State.")
title.chat("How is Title IX implementated in California State in terms of policies, plan, strategy, and other details.")