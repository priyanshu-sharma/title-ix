import autogen
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

config_list = [{"base_url": "http://0.0.0.0:8000", "model": "mistral"}]
llm_config = {
    "cache_seed": 42,  
    "temperature": 0,
    "config_list": config_list,
    "timeout": 120,
}

texas_assistant = RetrieveAssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant, which help in determining policies, plan and other implementation details of Title IX implementation in Texas.",
    llm_config=llm_config,
)

texas_ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    retrieve_config={
        "task": "qa",
        "docs_path": "https://raw.githubusercontent.com/priyanshu-sharma/title-ix/master/src/crawler/dataset_domain/output/texas.txt?token=GHSAT0AAAAAAB3JO6UQ4NKO7LH7E5NGO5QEZNGFFJA",
    },
)

texas_assistant.reset()
texas_ragproxyagent.initiate_chat(texas_assistant, problem="How is the Title IX Implementation in Texas?")