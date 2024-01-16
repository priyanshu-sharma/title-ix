import autogen

config_list = [{"base_url": "http://127.0.0.1:8080/v1", "api_key": "NULL"}]

# assistant = autogen.AssistantAgent("assistant", llm_config={"seed": 42, "config_list": config_list, "temperature": 0})
# user_proxy = autogen.UserProxyAgent(
#     "user_proxy",
#     llm_config={"config_list": config_list},
#     human_input_mode="NEVER",
#     max_consecutive_auto_reply=5,
#     is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
#     code_execution_config={
#         "work_dir": "coding",
#         "use_docker": False,  # set to True or image name like "python:3" to use docker
#     },
# )
# user_proxy.initiate_chat(
#     assistant,
#     message="""What date is today? Compare the year-to-date gain for META and TESLA.""",
# )

llm_config = {
    "cache_seed": 42,  
    "temperature": 0,
    "config_list": config_list,
    "timeout": 120,
}

user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="A human admin. Interact with the TexasLawyer and CaliforniaLawyer to discuss the differences in policies, plan and other implementation details of Title IX implementation. Policies, Plan and other implementation details needs to be approved as well as verified by this admin.",
    code_execution_config=False,
    human_input_mode="NEVER",
)

texas_lawyer = autogen.AssistantAgent(
    name="TexasLawyer",
    llm_config=llm_config,
    system_message="""Supreme Court Lawyer of the United States, representing Texas state. You need to defend the Texas state in this session of Supreme Court of the United States, which is primarily related to the policies, plan and other implementation details of Title IX implementation in Texas State.
    You need to provide some verified policies, laws that are pass by Texas Government in order to implement Title IX. You need to continuously defend, debate and interact with CaliforniaLawyer and Admin. Don't stop just after making the 1st point, continueously engage in a debate with CaliforniaLawyer.
""",
)

california_lawyer = autogen.AssistantAgent(
    name="CaliforniaLawyer",
    llm_config=llm_config,
    system_message="""Supreme Court Lawyer of the United States, representing California state. You need to defend the California state in this session of Supreme Court of the United States, which is primarily related to the policies, plan and other implementation details of Title IX implementation in California State.
    You need to provide some verified policies, laws that are pass by California Government in order to implement Title IX. You need to continuously defend, debate and interact with TexasLawyer and Admin. Don't stop just after making the 1st point, continueously engage in a debate with TexasLawyer.
""",
)

# engineer = autogen.AssistantAgent(
#     name="Engineer",
#     llm_config=gpt4_config,
#     system_message="""Engineer. You follow an approved plan. You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
# Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
# If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
# """,
# )
# scientist = autogen.AssistantAgent(
#     name="Scientist",
#     llm_config=gpt4_config,
#     system_message="""Scientist. You follow an approved plan. You are able to categorize papers after seeing their abstracts printed. You don't write code.""",
# )
# planner = autogen.AssistantAgent(
#     name="PolicyMaker",
#     system_message="""PolicyMaker. Suggest a policies, plans and other implementation details for the better Title IX implementation. 
# Revise the policies, plan and implementation details based on feedback from admin and critic, until admin approval.
# The plan may involve an engineer who can write code and a scientist who doesn't write code.
# Explain the plan first. Be clear which step is performed by an engineer, and which step is performed by a scientist.
# """,
#     llm_config=gpt4_config,
# )
# executor = autogen.UserProxyAgent(
#     name="Executor",
#     system_message="Executor. Execute the code written by the engineer and report the result.",
#     human_input_mode="NEVER",
#     code_execution_config={"last_n_messages": 3, "work_dir": "paper"},
# )
# critic = autogen.AssistantAgent(
#     name="Critic",
#     system_message="Critic. Double check plan, claims, policies and other minor details about the Title IX Implementation of both state from other agents and provide feedback. Check whether the plan includes adding verifiable info such as source URL.",
#     llm_config=gpt4_config,
# )
# agents=[user_proxy, engineer, scientist, planner, executor, critic], messages=[], max_round=50

groupchat = autogen.GroupChat(
    agents=[user_proxy, texas_lawyer, california_lawyer], messages=[], max_round=20
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

user_proxy.initiate_chat(
    manager,
    message="""
In this session of Supreme Court of the United States, we will try to formulate the differences in plans, policies, implementation and other details of the Title IX Implementation of two states - California, and Texas.
""",
)