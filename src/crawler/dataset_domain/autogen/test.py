import autogen

config_list = [{"base_url": "http://127.0.0.1:8080/v1", "api_key": "NULL"}]

assistant = autogen.AssistantAgent("assistant", llm_config={"seed": 42, "config_list": config_list, "temperature": 0})
user_proxy = autogen.UserProxyAgent(
    "user_proxy",
    llm_config={"config_list": config_list},
    human_input_mode="NEVER",
    max_consecutive_auto_reply=5,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,  # set to True or image name like "python:3" to use docker
    },
)
user_proxy.initiate_chat(
    assistant,
    message="""What date is today? Compare the year-to-date gain for META and TESLA.""",
)
