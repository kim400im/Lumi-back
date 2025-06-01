from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import List
from uuid import UUID
from datetime import datetime
from pprint import pprint
import json
from send_to_runpod_via_openai import run_llm_analysis, format_messages, find_similar_docs
from send_to_runpod_via_openai import supabase_client


app = FastAPI()

# 메시지 스키마
class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")  # ✅
    content: str

# 요청 본문 스키마
class ChatUploadRequest(BaseModel):
    user_id: UUID
    session_id: UUID
    character_name: str
    messages: List[Message]
    ended_at: datetime

# 응답 스키마
class ChatUploadResponse(BaseModel):
    status: str
    received_messages: int
    analysis: str  # 분석 결과도 같이 반환
    
    
@app.get("/ping")
def ping():
    print("✅ ping 받음")
    return {"message": "pong"}

@app.post("/api/chat-upload")
async def upload_chat_log(
    payload: ChatUploadRequest,
    request: Request,
    background_tasks: BackgroundTasks
):
    print("\n=== 📦 전체 Payload 수신 ===")
    print(json.dumps(payload.model_dump(), indent=2, ensure_ascii=False, default=str))
    print("===========================")

    if len(payload.messages) < 1:
        raise HTTPException(status_code=400, detail="메시지는 최소 10개 이상이어야 합니다.")

    # 분석 프롬프트 구성
    formatted_messages = format_messages([m.model_dump() for m in payload.messages])
    system_prompt = {"role": "system", "content": f"{payload.character_name}와의 대화 내용을 분석해줘"}
    full_prompt = [system_prompt] + formatted_messages
    
    print("📤 RunPod로 요청 전송 중...")  # 👈 이거 추가

    # Supabase: 분석 요청 저장
    request_res = supabase_client.table("analysis_requests").insert({
        "user_id": str(payload.user_id),
        "session_id": str(payload.session_id),
        "character_name": payload.character_name,
        "prompt_sent": json.dumps(full_prompt, ensure_ascii=False)
    }).execute()

    if not request_res.data:
        raise HTTPException(status_code=500, detail="요청 저장 실패")

    request_id = request_res.data[0]["id"]

    # 백그라운드 작업: 분석 요청 후 응답 저장
    background_tasks.add_task(analyze_and_store_response, request_id, full_prompt, payload.messages)

    return {"status": "accepted", "message": "분석은 백그라운드에서 수행됩니다."}

def analyze_and_store_response(request_id: UUID, full_prompt: List[dict], messages: List[Message]):
    try:
        # 1. 유저 대화 문자열로 합치기
        dialogue_text = "\n".join([f"{m['role']}: {m['content']}" for m in full_prompt[1:]])

        # 2. FAISS 유사 문서 검색
        similar_chunks = find_similar_docs(dialogue_text)
        
        # 기존 대화 내용 + RAG 문서를 결합한 프롬프트 구성
        final_prompt = [
            {"role": "system", "content": "당신은 사용자의 대화 내용을 보고, 외부 문서를 참고하여 위험 여부나 특이 사항을 분석하는 전문가입니다."},
            *format_messages([m.model_dump() for m in messages]),
            {"role": "user", "content": "이 대화 내용을 분석해줘. 참고 문서:\n\n" + "\n\n".join(similar_chunks)}
        ]
        
        
        response_text = run_llm_analysis(messages=final_prompt, character_name="")
        
        # RunPod로 분석 요청
        # 먼저 요청 시 저장된 user_id를 가져옴
        request_row = supabase_client.table("analysis_requests").select("user_id").eq("id", str(request_id)).single().execute()
        user_id = request_row.data["user_id"]


        # Supabase: 결과 저장
        supabase_client.table("analysis_results").insert({
            "request_id": str(request_id),
            "user_id": user_id,  # ✅ 여기 추가
            "llm_response": response_text
        }).execute()

        cleaned_text = response_text.encode("utf-8", "replace").decode("utf-8")
        print(f"📊 분석 결과 저장 완료: {cleaned_text[:50]}...")
    except Exception as e:
        print(f"❌ 분석 실패: {str(e)}")
