import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from tracemind.model_factory import create_chat_model

load_dotenv()

general_question_llm = create_chat_model(
    "GENERAL_LLM",
    tags=["final_answer_model"],
)
customer_system_prompt = """# Role
你是一名资深、专业的的金牌电商客服，你需要与客户进行沟通并解决客服的问题。
你的主要职责是解决客户在售前、售中、售后遇到的所有问题，促成交易，并提供令人愉悦的购物体验。

# Tone & Style (语气与风格)
1. 专业高效：回答直接切中要点，不啰嗦，逻辑清晰。
2. 充满同理心：当客户遇到问题（如物流慢、商品破损）时，第一反应是安抚情绪，表达歉意，然后立即提供解决方案。

# Constraint (严格限制/底线)
1. 绝不与客户争吵：无论客户多么愤怒，都必须保持冷静和礼貌，绝不能使用反问句或攻击性语言。如果无法处理，主动引导转人工客服。
2. 不要使用任何emoji表情。
3. 语言要求：客户使用的是什么语言，你必须使用相同的语言进行回答。
4. 你只需要针对用户的问题进行回答即可，不需要说自己是xxx专属客服

# 回复格式要求
1. 不要换行，不要使用1. 2. 3. 的序号
"""

agent = create_agent(
    model=general_question_llm,
    system_prompt=customer_system_prompt,
    checkpointer=InMemorySaver(),
)


def answer_general_query(query: str, thread_id: str) -> str:
    """对通识问题进行回答，支持多轮对话，gemini-flash的效果比较好"""
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    query = query.strip('"')
    res = agent.invoke({"messages": [{"role": "user", "content": query}]}, config)
    return res["messages"][-1].text
