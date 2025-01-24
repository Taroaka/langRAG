import chainlit as cl
from chainlit.input_widget import Select

from src.llm_response import generate_message

# ========== 1. ChatProfileで「単一 or エージェント」選択 ==========
@cl.set_chat_profiles
async def chat_profile():
    """
    画面上部に表示される「ChatProfile」選択を定義。
    ユーザーがここで「single」(単一呼び出し)か「agent」(複数ステップ)かを選択する。
    """
    return [
        cl.ChatProfile(
            name="通常",
            markdown_description="**通常のAI**: モデルの選択は歯車アイコンから。",
            icon="https://i.imgur.com/vhHfHih.png",
        ),
        cl.ChatProfile(
            name="改善エージェント",
            markdown_description="**改善エージェント**: 2ステップ解答で出力微増",
            icon="https://i.imgur.com/zJgxfum.jpeg"
        ),
        cl.ChatProfile(
            name="langchainマスター",
            markdown_description="**エージェント呼び出し**: LangChainの最新公式データを取得するAIを呼び出します",
            icon="https://i.imgur.com/zJgxfum.jpeg"
        ),
        cl.ChatProfile(
            name="オーケストラエージェント",
            markdown_description="**エージェント呼び出し**: LangGraphを使い複数ステップでAIを呼び出します",
            icon="https://i.imgur.com/zJgxfum.jpeg"
        )
    ]

# ========== 2. on_chat_start でLLMモデルのみSelect ==========
@cl.on_chat_start
async def start():
    """
    Chatを開始したタイミングで一度だけ呼ばれる。
    UI上に「AIモデル」のセレクトだけ出す。
    （単一orエージェントの分岐はChatProfileで既に選択される）
    """
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="AI モデル",
                items={
  "Gemini Exp 1121 Free": "google/gemini-exp-1121:free",
                    "deepseekV3":"deepseek/deepseek-chat",
                      "Deepseek R1": "deepseek/deepseek-r1",
  "Claude 3.5 Sonnet Beta": "anthropic/claude-3.5-sonnet:beta",
                    "GPT-4o": "openai/gpt-4o-2024-08-06",
                    "Mistral 7B Instruct": "mistralai/mistral-7b-instruct",
                    "o1":"openai/o1",
                    "Claude 3.5 Sonnet": "anthropic/claude-3.5-sonnet",
                    "Gemini Pro 1.5": "google/gemini-pro-1.5",
                },
                initial_value="google/gemini-exp-1121:free",
            )
        ]
    ).send()

    value = settings["Model"]

    await update_settings(value)

@cl.on_settings_update
async def update_settings(settings):
    """
    チャット中に設定が変わった際に呼ばれる。
    （念のためモデル選択を再セットするだけ）
    """
    cl.user_session.set("model_name",settings)

# ========== 3. on_messageでユーザーの入力を受け取り、generate_messageに丸投げ ==========
@cl.on_message
async def main(message: cl.Message):
    """
    ユーザーからメッセージが送られたら実行される処理。
    ChatProfile(= single or agent)とModelをsessionから取得し、
    generate_messageに渡してレスポンスを返す。
    """
    usage_profile = cl.user_session.get("chat_profile")  # single or agent
    model_name = cl.user_session.get("model_name") # 選択したLLMモデル

    user_text = message.content

    # LangGraphで単一 or 複数ステップ呼び出しを行う処理をgenerate_messageにまとめる
    final_response = await generate_message(
        user_text=user_text,
        usage_profile=usage_profile,
        model_name=model_name
    )

    # ChainlitのUIに返す
    await cl.Message(content=final_response).send()
