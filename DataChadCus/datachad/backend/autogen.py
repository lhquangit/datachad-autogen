from typing_extensions import Annotated
import chromadb

import autogen
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
import os
# import logger

def load_config():
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    file_path = os.path.join(parent_dir, "config.json")

    # import ipdb; ipdb.set_trace(context=10)

    config_list = autogen.config_list_from_json(file_path)

    llm_config = {
    "timeout": 60,
    "temperature": 0,
    "config_list": config_list,
    }
    # logger.info(f"Autogen data path: {file_path}")
    # logger.info(f"LLM model: {[config_list[i]["model"] for i in range(len(config_list))]}")
    # import ipdb; ipdb.set_trace(context=10)
    return config_list, llm_config

def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

def create_agents(config_list, llm_config):
    # config_list, llm_config = load_config()
    boss = autogen.UserProxyAgent(
                                name="Boss",
                                is_termination_msg=termination_msg,
                                human_input_mode="NEVER",
                                code_execution_config=False,  # we don't want to execute code in this case.
                                default_auto_reply="Reply `TERMINATE` if the task is done.",
                                description="The boss who ask questions and give tasks.",
                                )
    
    boss_aid = RetrieveUserProxyAgent(
                                name="Boss_Assistant",
                                is_termination_msg=termination_msg,
                                human_input_mode="NEVER",
                                max_consecutive_auto_reply=3,
                                retrieve_config={
                                    "task": "code",
                                    "docs_path": "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md",
                                    "chunk_token_size": 1000,
                                    "model": config_list[0]["model"],
                                    "client": chromadb.PersistentClient(path="/tmp/chromadb"),
                                    "collection_name": "groupchat",
                                    "get_or_create": True,
                                },
                                code_execution_config=False,  # we don't want to execute code in this case.
                                description="Assistant who has extra content retrieval power for solving difficult problems.",
                                )
    
    coder = AssistantAgent(
                            name="Senior_Python_Engineer",
                            is_termination_msg=termination_msg,
                            system_message="You are a senior python engineer, you provide python code to answer questions. Reply `TERMINATE` in the end when everything is done.",
                            llm_config=llm_config,
                            description="Senior Python Engineer who can write code to solve problems and answer questions.",
                        )
    
    pm = autogen.AssistantAgent(
                            name="Product_Manager",
                            is_termination_msg=termination_msg,
                            system_message="You are a product manager. Reply `TERMINATE` in the end when everything is done.",
                            llm_config=llm_config,
                            description="Product Manager who can design and plan the project.",
                        )
    
    reviewer = autogen.AssistantAgent(
                            name="Code_Reviewer",
                            is_termination_msg=termination_msg,
                            system_message="You are a code reviewer. Reply `TERMINATE` in the end when everything is done.",
                            llm_config=llm_config,
                            description="Code Reviewer who can review the code.",
                        )
    
    return boss, coder, pm, reviewer


def reset_agents(boss, coder, pm, reviewer):
    boss.reset()
    coder.reset()
    pm.reset()
    reviewer.reset()


