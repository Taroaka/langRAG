import os
from dotenv import load_dotenv
from typing import Optional
import openai

# LangGraph関連
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.schema.runnable.config import RunnableConfig

# from single_call import single_call
from src.improve_agent import build_agent_graph,single_call
from src.langchain_git_rag import process_query
load_dotenv()


async def generate_message(user_text: str, usage_profile: str, model_name: str) -> str:
    """
    - usage_profile='single'ならChatCompletionを1度だけ呼び出す
    - usage_profile='agent'ならLangGraphで2ステップ呼び出す

    いずれにせよ最終的に返すのは最終ステップの回答文字列
    """

    if usage_profile == "single":
        # 単一呼び出し
        return single_call(user_text, model_name)

    elif usage_profile == "agent":
        # LangGraphで2ステップ
        state = {
            "messages": [],
            "user_text": user_text,
            "model_name": model_name
        }
        AGENT_GRAPH = build_agent_graph()
        # graph.astream でストリーミングしたいなら async for 
        # ただ最終文字列だけ取るなら .invoke() でもOK
        output = []
        async for msg_out, metadata in AGENT_GRAPH.astream(state, stream_mode="messages", config=RunnableConfig()):
            if isinstance(msg_out, AIMessage):
                output.append(msg_out.content)

        # 最後に生成されたAIMessageのコンテンツを返す
        return output[-1] if output else "申し訳ありません、何らかのエラーが発生しました。"
    
    elif usage_profile == "agent":
        process_query(user_text, model_name)

    else:
        # fallback
        return "使用するプロファイルが無効です。"
