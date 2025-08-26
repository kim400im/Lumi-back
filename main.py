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
    
# api
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
        
        # 개선된 상세 프롬프트 구성
        system_prompt = """당신은 온라인 대화 내용을 분석하여 범죄 가능성이나 위험 요소를 탐지하는 전문 분석가입니다.
        
다음 대화 내용을 분석하고, 아래 항목들을 중심으로 상세하게 평가해주세요:

## 분석 기준
1. **범죄 위험도 평가**
   - 직접적 범죄 언급 (협박, 폭행, 성범죄, 사기 등)
   - 간접적 범죄 암시 (은밀한 표현, 암호화된 언어)
   - 불법 물질/서비스 거래 징후

2. **심리적 위험 신호**
   - 자해/자살 의도나 암시
   - 타인에 대한 해악 의도
   - 정신적 불안정 징후

3. **취약계층 대상 위험**
   - 미성년자 대상 부적절한 접근
   - 노인/장애인 대상 사기나 학대
   - 사회적 약자 대상 착취

4. **기타 주의사항**
   - 개인정보 유출 위험
   - 사생활 침해 가능성
   - 스토킹이나 추적 행위

## 분석 예시
**높은 위험도 예시:**
- "오늘 밤에 만나서 그 일을 처리하자"
- "돈이 필요하면 이 방법이 있어"
- "아무도 모르게 할 수 있어"

**중간 위험도 예시:**
- 과도한 개인정보 요구
- 반복적인 만남 요청
- 금전 관련 대화

**낮은 위험도 예시:**
- 일상적인 대화
- 건전한 관심사 공유
- 정상적인 사회적 교류

## 출력 형식
다음 형식으로 **반드시 2000토큰 이상의 상세한 분석**을 제공해주세요:

**위험도 등급:** [높음/중간/낮음]

**주요 발견사항:**
- 구체적인 위험 요소들을 상세히 나열
- 해당 부분의 대화 내용 인용
- 위험도 판단 근거 설명

**세부 분석:**
- 대화 패턴 분석
- 언어 사용 특성
- 시간대/빈도 패턴
- 상대방 반응 분석

**권장 조치사항:**
- 즉시 필요한 조치
- 장기적 모니터링 방안
- 관련 기관 신고 필요성

**추가 모니터링 포인트:**
- 향후 주의깊게 관찰할 요소들
- 패턴 변화 감지 포인트

반드시 구체적인 근거와 함께 상세한 분석을 제공하고, 분석 결과가 2000토큰 이상이 되도록 충분히 자세하게 작성해주세요."""

        
        # 기존 대화 내용 + RAG 문서를 결합한 프롬프트 구성
        final_prompt = [
            {"role": "system", "content": system_prompt},
            *format_messages([m.model_dump() for m in messages]),
            {"role": "user", "content": f"""위의 대화 내용을 분석해주세요.

참고 문서:
{chr(10).join(similar_chunks)}

위의 시스템 프롬프트에 명시된 형식과 기준에 따라 매우 상세하고 구체적인 분석을 제공해주세요. 
분석 결과는 반드시 2000토큰 이상의 길이로 작성하고, 각 항목별로 구체적인 근거와 예시를 포함해주세요."""}
        ]

        # RunPod로 분석 요청 (토큰 길이 증가)
        response_text = run_llm_analysis(
            messages=final_prompt, 
            character_name="",
            max_tokens=3000,  # 토큰 길이 증가
            temperature=0.3   # 일관성 있는 분석을 위해 낮은 temperature
        )

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
