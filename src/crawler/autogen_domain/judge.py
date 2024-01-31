import autogen
from autogen.agentchat.contrib.agent_builder import AgentBuilder

config_file_or_env = 'OAI_CONFIG_LIST'
default_llm_config = {'temperature': 0}

builder = AgentBuilder(config_file_or_env=config_file_or_env, builder_model='ollama/mistral', agent_model='ollama/mistral', host='localhost', max_agents=5)
building_task = """We will mainly discuss the best policies and shortcomings of the Implementation of Title IX of various states such as California, and Texas, and then prepare a report about the its implementation, policies, and other details in different states.
               Each state representative will present their own states policies and shortcomings. (Use on state government data from their official websites)"""
agent_list, agent_configs = builder.build(building_task, default_llm_config, coding=False)

def start_task(execution_task: str, agent_list: list, llm_config: dict):
    config_list = autogen.config_list_from_json(config_file_or_env)
    group_chat = autogen.GroupChat(agents=agent_list, messages=[], max_round=20, speaker_selection_method="round_robin")
    manager = autogen.GroupChatManager(
        groupchat=group_chat, llm_config={"config_list": config_list, **llm_config}, code_execution_config={"use_docker": False},
        system_message="""You are the Conversation Initiator and Moderator. Your role is to provide a platform for different states representative to engage in a discussion. You are responsible for introducing topics of Implementation of Title IX in California, and Texas, and moderating the conversation to ensure it remains on track and constructive.
                        You monitor the conversation for relevance and adherence to the topic, intervening if necessary. You encourage balance and ensure all the state representative have equal opportunities to express their viewpoints. Your communication style is neutral, concise, and clear.
                        Directive: Facilitate a smooth and engaging conversation between various states representatives without imposing any bias.""",
    )
    agent_list[0].initiate_chat(manager, message=execution_task)

start_task(
    execution_task="Find all the common and different ideas about the implementation of Title IX in California, and Texas. Give alternative chance to both side in order to defend themselves and you can also read other agents conversation. Discussion should be more on differences.",
    agent_list=agent_list,
    llm_config=default_llm_config
)

a = ''
for agent in agent_list:
    contents = agent.chat_messages.values()
    for content in contents:
        for data in content:
            a = a + data['content'] + '\n\n'

with open('policy.txt', "w") as f:
    f.write(a)