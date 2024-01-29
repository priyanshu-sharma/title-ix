import autogen
from autogen.agentchat.contrib.agent_builder import AgentBuilder


# class TitleAgent:
#     def __init__(self):
#         self.default_llm_config = {"temperature": 0}
#         self.builder = AgentBuilder(
#             config_file_or_env="OAI_CONFIG_LIST",
#             builder_model='ollama/mistral',
#             agent_model='ollama/mistral',
#             host='localhost',
#             endpoint_building_timeout=1000,
#         )

#     def start_task(self, execution_task):
#         config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")
#         group_chat = autogen.GroupChat(agents=self.agent_list, messages=[], max_round=12)
#         manager = autogen.GroupChatManager(
#             groupchat=group_chat,
#             llm_config={"config_list": config_list, **self.default_llm_config},
#             system_message="""You are the Conversation Initiator and Moderator. Your role is to provide a platform for different states representative to engage in a discussion. You are responsible for introducing topics of Implementation of Title IX in California, Texas, Utah, and in New York and moderating the conversation to ensure it remains on track and constructive.  
#                         You monitor the conversation for relevance and adherence to the topic, intervening if necessary. You encourage balance and ensure all the state representative have equal opportunities to express their viewpoints. Your communication style is neutral, concise, and clear.
#                         Directive: Facilitate a smooth and engaging conversation between various states representatives without imposing any bias.""",
#             code_execution_config={"use_docker": False},
#             default_auto_reply="That's very interesting. Tell me more.",
#         )
#         self.agent_list[0].initiate_chat(manager, message=execution_task)

#     def execute_task(self, building_task, execution_task):
#         self.agent_list, agent_configs = self.builder.build(
#             building_task, self.default_llm_config, coding=False, code_execution_config=None
#         )
#         self.start_task(execution_task=execution_task)
#         saved_path = self.builder.save()


# ta = TitleAgent()
# building_task = "Find the current status of Implementation of Title IX in various states such as California, Texas, Utah, New York, etc, by crawling the web, and then prepare a report about the its implementation, policies, and other details in different states. Do not write any"
# execution_task = (
#     "Find all the common and different ideas about the implementation of Title IX in California, Texas, Utah and in New York. Give alternative chance to both side in order to defend themselves and you can also read other agents conversation.",
# )
# ta.execute_task(building_task, execution_task)


config_file_or_env = 'OAI_CONFIG_LIST'
default_llm_config = {'temperature': 0}

builder = AgentBuilder(config_file_or_env=config_file_or_env, builder_model='ollama/mistral', agent_model='ollama/mistral', host='0.0.0.0', max_agents=5)
building_task = """In today's session of Supreme Court of the United States, we will mainly discuss the best policies and shortcomings of the Implementation of Title IX of various states such as California, Texas, Utah, New York, etc, and then prepare a report about the its implementation, policies, and other details in different states.
                In this session, lawyer representing various states will present their own states policies and shortcoming in order to formulate the nation-wide policy for the same. (Use on state government data from their official websites)"""
agent_list, agent_configs = builder.build(building_task, default_llm_config, coding=False)

def start_task(execution_task: str, agent_list: list, llm_config: dict):
    config_list = autogen.config_list_from_json(config_file_or_env)
    group_chat = autogen.GroupChat(agents=agent_list, messages=[], max_round=40)
    manager = autogen.GroupChatManager(
        groupchat=group_chat, llm_config={"config_list": config_list, **llm_config}, code_execution_config={"use_docker": False},
        system_message="""You are the Conversation Initiator and Moderator. Your role is to provide a platform for different states representative to engage in a discussion. You are responsible for introducing topics of Implementation of Title IX in California, Texas, Utah, and in New York and moderating the conversation to ensure it remains on track and constructive.  
                        You monitor the conversation for relevance and adherence to the topic, intervening if necessary. You encourage balance and ensure all the state representative have equal opportunities to express their viewpoints. Your communication style is neutral, concise, and clear.
                        Directive: Facilitate a smooth and engaging conversation between various states representatives without imposing any bias.""",
    )
    agent_list[0].initiate_chat(manager, message=execution_task)

start_task(
    execution_task="Find all the common and different ideas about the implementation of Title IX in California, Texas, Utah and in New York. Give alternative chance to both side in order to defend themselves and you can also read other agents conversation.",
    agent_list=agent_list,
    llm_config=default_llm_config
)
builder.save('new_judge.json')