from openai import OpenAI
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from supabase import create_client, Client
from dotenv import load_dotenv

# 📥 환경 변수 로드
load_dotenv()

# Supabase 설정
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Embedding 모델
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

faiss_db = FAISS.load_local(
    "faiss_db",
    embedding_model,
    allow_dangerous_deserialization=True  # ✅ 반드시 명시
)

# RunPod 설정
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
MODEL_NAME = os.getenv("RUNPOD_MODEL_NAME")

client = OpenAI(
    api_key=RUNPOD_API_KEY,
    base_url=f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/openai/v1",
)

def format_messages(messages: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in messages]

def run_llm_analysis(messages: list[dict], character_name: str) -> str:
    print("🔍 [LLM 분석 시작]")
    print(f"🧠 캐릭터 이름: {character_name}")
    print(f"🗒️ 총 메시지 수: {len(messages)}")
    
    formatted_messages = [{"role": "system", "content": f"{character_name}와의 대화 내용을 분석해줘"}] + format_messages(messages)
    print("📨 보낼 메시지 목록:")
    for msg in formatted_messages:
        print(f" - ({msg['role']}) {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=formatted_messages,
            temperature=0.7,
            max_tokens=512,
        )

        print("✅ RunPod 응답 수신 완료")
        if response and hasattr(response, 'choices') and len(response.choices) > 0:
            print("💬 분석 결과:")
            print(response.choices[0].message.content.strip())
            return response.choices[0].message.content.strip()

        print("⚠️ RunPod 응답은 정상적이지만 분석 내용 없음")
        return "❌ LLM 응답 없음"
    
    except Exception as e:
        print(f"❌ 예외 발생: {str(e)}")
        return f"❌ RunPod 요청 실패: {str(e)}"


def find_similar_docs(query: str, k: int = 3) -> list[str]:
    results = faiss_db.similarity_search(query, k=k)
    docs = [doc.page_content for doc in results]
    
    # # ✅ 여기 로그 추가
    # print("🔎 FAISS 검색 결과 문서:")
    # for i, chunk in enumerate(docs):
    #     print(f"  [{i+1}] {chunk[:100]}{'...' if len(chunk) > 100 else ''}")
    
    # return docs
    print("🔎 FAISS 검색 결과 문서 (전체 출력):")
    for i, chunk in enumerate(docs):
        print(f"\n===== [문서 {i+1}] =====")
        print(chunk)  # ✅ 전체 내용 출력
    
    return docs