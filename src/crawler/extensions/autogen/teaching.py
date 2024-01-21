import autogen
from autogen import ConversableAgent, UserProxyAgent
from autogen.agentchat.contrib.capabilities.teachability import Teachability

config_list = [{"base_url": "http://0.0.0.0:8000", "model": "mistral"}]
llm_config = {
    "cache_seed": 42,  
    "temperature": 0,
    "config_list": config_list,
    "timeout": 120,
}


teachable_agent = ConversableAgent(
    name="teachable_agent",
    llm_config=llm_config,
)

teachability = Teachability(
    verbosity=0,
    reset_db=True,
    path_to_db_dir="./output",
    recall_threshold=1.5, 
)

teachability.add_to_agent(teachable_agent)

user = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False,
    max_consecutive_auto_reply=3,
)

text = "How is Title IX implemented in California?"
user.initiate_chat(teachable_agent, message=text, clear_history=True)

text = "How is Title IX implemented in Texas?"
user.initiate_chat(teachable_agent, message=text, clear_history=True)

text = "How is the implementation of Title IX is different in Texas as compared to California and vice vera?"
user.initiate_chat(teachable_agent, message=text, clear_history=True)