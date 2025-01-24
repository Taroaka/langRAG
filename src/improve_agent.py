import os
from dotenv import load_dotenv
from typing import Optional
import openai

# RAG関連
from langchain_community.document_loaders import GitLoader

# LangGraph関連
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.schema.runnable.config import RunnableConfig

# from single_call import single_call,chatcompletion_create

SYSTEM_CONTENT = """あなたはフレンドリーなアシスタントです。質問に答えて下さい。"""

def file_filter(file_path:str) -> bool:
    return file_path.endswith(".mdx") 

def single_call(user_text: str, model_name: str) -> str:
    """
    ChatCompletionを1回だけ呼び出す
    """
    messages = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user",   "content": user_text},
    ]
    return chatcompletion_create(messages, model_name)

def chatcompletion_create(messages, model_name):
    try:
        loader = GitLoader(
            clone_url = "https://github.com/langchain-ai/langchain",
            repo_path="./langchain",
            branch ="master",
            file_filter = file_filter
        )

        raw_docs = loader.load()

        print(len(raw_docs))

        client = openai.OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"), 
            base_url="https://openrouter.ai/api/v1"
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating message: {e}")
        return "申し訳ありませんが、応答を生成できませんでした。"



def agent_step1(state: MessagesState):
    user_text = state.get("user_text", "")
    model_name = state.get("model_name", "openai/gpt-4o-2024-08-06")

    content = single_call(user_text, model_name)
    return {"messages": [AIMessage(content=content)]}

def agent_step2(state: MessagesState):
    """
    直前のAIMessageを再度「より良くして」と呼び出す
    """
    user_text = state.get("user_text", "")
    model_name = state.get("model_name", "openai/gpt-4o-2024-08-06")
    last_ai_msg = state["messages"][-1]
    if not isinstance(last_ai_msg, AIMessage):
        raise ValueError("直前メッセージがAIMessageではありません")

    improved_content = chatcompletion_create([
        {"role": "system", "content": "あなたは文章校正アシスタントです。"},
        {"role": "user", "content": f"以下の文章をより良くしてください:\n\n{last_ai_msg.content}"}
    ], model_name)

    return {"messages": [AIMessage(content=improved_content)]}

def build_agent_graph():
    """
    2ステップ処理のStateGraphを返す
    """
    builder = StateGraph(MessagesState)

    builder.add_node("step1", agent_step1)
    builder.add_node("step2", agent_step2)

    builder.add_edge(START, "step1")
    builder.add_edge("step1", "step2")
    builder.add_edge("step2", END)

    return builder.compile()