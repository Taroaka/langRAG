import os
from dotenv import load_dotenv
from typing import Optional
import openai

load_dotenv()

# ChatCompletionのsystemプロンプト例
SYSTEM_CONTENT = """あなたはフレンドリーなアシスタントです。質問に答えて下さい。"""

# === 1) 単一呼び出し ===
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

