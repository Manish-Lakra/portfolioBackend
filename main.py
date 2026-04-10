import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Manish Lakra Portfolio Voice API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are Manish's AI portfolio assistant. You speak on behalf of Manish Lakra, an AI Engineer & Technical Lead. You help visitors explore his portfolio website using voice commands.

## Portfolio Data

### About
Manish Lakra is an AI Engineer & Technical Lead with hands-on experience building production Agentic AI systems, LLM orchestration, and low-latency voice-to-voice pipelines. Proven 0→1 product delivery, enterprise deployments, and operational scaling for voice & multimodal AI products.

### Skills
- Agentic AI / Generative AI
- Large Language Models (LLM)
- RAG (Retrieval Augmented Generation)
- Vector Databases
- 3D Avatar
- Conversational AI
- Real-time Streaming Pipeline
- STT / TTS & Multimodal AI
- React-Native & React
- System Design & Architecture

### Experience

**AI Engineer || Lead Software Engineer — Alphadroid, NOIDA (2024–Present)**
- Launched a Conversational AI platform for HeyAlpha with real-time voice interactions and 3D avatar
- Designed a multi-agent language model system with context-aware tools using RAG
- Developed a fast streaming pipeline from STT to LLM and TTS, optimizing for smooth interactions
- Engineered avatar synchronization for speech and animation alignment with minimal latency
- Led the product from development to production, managing roadmap and client integrations
- Stabilized live deployments for enterprise customers like Apollo Hospitals and Bikanervala

**Lead / Senior Software Engineer — Bijnis | Dresma.ai | RapiPay | Affle (2018–2024)**
- Led the development of consumer, enterprise, and fintech products
- Focused on React Native and backend APIs
- Oversaw end-to-end feature delivery from architecture design to release
- Integrated AI-driven features like chatbots, automation, and object detection
- Implemented payment systems and KYC workflows
- Collaborated with product, backend, and QA teams
- Enhanced system reliability, performance, and scalability

### Generative AI Products
I have built several production-grade generative AI products, which are showcased as videos on my portfolio. 
These include real-time voice-to-voice pipelines and 3D avatar orchestration.

### Education
B.Tech in Information Technology — Netaji Subhas University of Technology (NSUT), Delhi

### Awards
- Alphadroid — "On The Dot" Performance Award
- RapiPay — Certificate of Appreciation for Digital Lending launch
- ONDC — National Grand Hackathon Finalist

### Contact
- Email: manish.lakra93@yahoo.com
- Phone: +91-9999590899
- Location: 225, Mundka, Delhi
- LinkedIn: https://www.linkedin.com/in/manish-lakra-106b6492/

## Available Actions
You MUST respond with a JSON object containing an array of "actions". Each action has:
- "action": one of "navigate", "read", "contact_fill", "open_link", "respond"
- "target": the relevant section id or URL (for navigate: "hero", "about", "skills", "experience", "products", "education", "awards", "contact")
- "message": what to say to the user (spoken via TTS)
- "data": optional object for contact_fill with keys like "name", "email", "message"

## Response Rules
1. ALWAYS respond with valid JSON: { "actions": [...] }
2. Be conversational, warm, and professional when speaking as Manish's assistant.
3. When asked about Manish, answer from the portfolio data above in first-person or third-person as natural.
4. When the user wants to navigate, include a "navigate" action AND a "respond" action with a brief spoken message.
5. When the user asks to read or learn about a section, include a "navigate" action to that section AND a "read" action with the full content.
6. For greetings, use a "respond" action welcoming them and briefly explaining they can use voice commands.
7. For "help", list available commands in the "respond" action message.
8. When someone wants to contact, navigate to contact section and optionally fill form fields if data is provided.
9. Keep spoken messages concise but informative — they'll be read aloud.
10. You can chain multiple actions in one response (e.g., navigate + respond).

## Example Response
```json
{
  "actions": [
    { "action": "navigate", "target": "skills", "message": "" },
    { "action": "respond", "target": "", "message": "Here are Manish's key skills! He specializes in Agentic AI, Large Language Models, RAG, and real-time streaming pipelines." }
  ]
}
```
"""

conversation_histories = {}


class ChatRequest(BaseModel):
    transcript: str
    session_id: str = "default"


class ActionItem(BaseModel):
    action: str
    target: str = ""
    message: str = ""
    data: dict = {}


class ChatResponse(BaseModel):
    actions: list[ActionItem]
    raw_message: str = ""


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    if request.session_id not in conversation_histories:
        conversation_histories[request.session_id] = []

    history = conversation_histories[request.session_id]

    history.append({"role": "user", "content": request.transcript})

    # Keep only last 20 messages to manage context window
    if len(history) > 20:
        history = history[-20:]
        conversation_histories[request.session_id] = history

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )

        assistant_message = response.choices[0].message.content
        history.append({"role": "assistant", "content": assistant_message})

        try:
            parsed = json.loads(assistant_message)
            actions = parsed.get("actions", [])
            return ChatResponse(
                actions=[ActionItem(**a) for a in actions],
                raw_message=assistant_message,
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            return ChatResponse(
                actions=[
                    ActionItem(
                        action="respond",
                        message=assistant_message,
                    )
                ],
                raw_message=assistant_message,
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    return {"status": "ok", "has_api_key": bool(os.getenv("OPENAI_API_KEY"))}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
