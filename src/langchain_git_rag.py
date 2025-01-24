from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv
import openai

load_dotenv()

# ChatCompletionのsystemプロンプト例
SYSTEM_CONTENT = """あなたはフレンドリーなアシスタントです。質問に答えて下さい。"""

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

def single_call(user_text: str, model_name: str) -> str:
    """
    ChatCompletionを1回だけ呼び出す
    """
    messages = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user",   "content": user_text},
    ]
    return chatcompletion_create(messages, model_name)

def process_query(query: str, model_name: str) -> str:
    def file_filter(file_path: str) -> bool:
        return file_path.endswith(".mdx")

    # Gitリポジトリからデータをロード
    loader = GitLoader(
        clone_url="https://github.com/langchain-ai/langchain",
        repo_path="./langchain",
        branch="master",
        file_filter=file_filter,
    )
    raw_docs = loader.load()

    # ドキュメントを分割
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(raw_docs)

    # クエリを埋め込みベクトル化
    embeddings = OpenAIEmbeddings()

    vector = embeddings.embed_query(query)

    # Chromaデータベースを作成し検索
    db = Chroma.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    context_docs = retriever.invoke(query)

    if context_docs:
        first_doc = context_docs[0]
        page_content = first_doc.page_content
        return single_call(page_content, model_name)
    else:
        return "関連する情報が見つかりませんでした。"
