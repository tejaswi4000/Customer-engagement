from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Customer Engagement LLM (RAG)")

# ----------------------------
# Embeddings + Vector Store
# ----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # small + fast

DOCS: List[Dict[str, Any]] = [
    {
        "id": "refund_policy",
        "text": "Refunds are available within 30 days of delivery for unused items. "
                "For damaged items, submit photos within 48 hours."
    },
    {
        "id": "shipping_policy",
        "text": "Standard shipping is 3-5 business days. Expedited shipping is 1-2 business days."
    },
    {
        "id": "order_tracking",
        "text": "You can track your order using the tracking link sent by email after shipment."
    },
    {
        "id": "warranty",
        "text": "Warranty covers manufacturing defects for 1 year. It does not cover accidental damage."
    },
]

doc_vectors = embedder.encode([d["text"] for d in DOCS], convert_to_numpy=True).astype("float32")
dim = doc_vectors.shape[1]
index = faiss.IndexFlatIP(dim)  # cosine similarity if vectors are normalized
# Normalize for cosine
faiss.normalize_L2(doc_vectors)
index.add(doc_vectors)

def retrieve(query: str, k: int = 3) -> List[Dict[str, Any]]:
    qv = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(qv)
    scores, ids = index.search(qv, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        results.append({"score": float(score), **DOCS[idx]})
    return results

# ----------------------------
# LLM Client (plug-in)
# ----------------------------
class LLMClient:
    """
    Replace generate() with your provider call.
    Keep this interface stable so you can swap OpenAI/Azure/Bedrock/Vertex/local.
    """
    def generate(self, system: str, user: str) -> str:
        # TODO: Replace with real LLM call
        # For now: simple placeholder
        return (
            "I can help with that. Based on our policy: " +
            user[:200] + " ..."
        )

llm = LLMClient()

# ----------------------------
# Customer Engagement Logic
# ----------------------------
INTENT_LABELS = ["order_status", "refund", "billing", "technical", "product_info", "other"]

def classify_intent(message: str) -> str:
    m = message.lower()
    if "track" in m or "tracking" in m or "order" in m:
        return "order_status"
    if "refund" in m or "return" in m:
        return "refund"
    if "charge" in m or "invoice" in m or "payment" in m:
        return "billing"
    if "error" in m or "not working" in m or "issue" in m:
        return "technical"
    if "feature" in m or "price" in m or "does it" in m:
        return "product_info"
    return "other"

def sentiment_hint(message: str) -> str:
    m = message.lower()
    if any(w in m for w in ["angry", "frustrated", "upset", "terrible", "worst"]):
        return "negative"
    if any(w in m for w in ["thanks", "great", "awesome", "love"]):
        return "positive"
    return "neutral"

def next_best_action(intent: str, sentiment: str) -> Dict[str, Any]:
    # Simple rule-based NBA (you can replace with a model later)
    if sentiment == "negative":
        return {"action": "escalate_to_human", "priority": "high", "reason": "customer frustration detected"}
    if intent == "refund":
        return {"action": "offer_resolution_steps", "priority": "medium", "reason": "refund intent"}
    if intent == "order_status":
        return {"action": "provide_tracking_steps", "priority": "low", "reason": "order status intent"}
    if intent == "product_info":
        return {"action": "recommend_product", "priority": "low", "reason": "product inquiry"}
    return {"action": "general_assist", "priority": "low", "reason": "default"}

def build_grounded_prompt(customer_message: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    context = "\n".join([f"- ({d['id']}) {d['text']}" for d in retrieved_docs])
    return f"""
Customer message:
{customer_message}

Use ONLY the following company info to answer (don’t invent):
{context}

Write a helpful, short response in plain English.
If the info is missing, ask 1 clarification question.
""".strip()

# ----------------------------
# API
# ----------------------------
class ChatRequest(BaseModel):
    customer_id: str
    message: str
    channel: Optional[str] = "webchat"

class ChatResponse(BaseModel):
    intent: str
    sentiment: str
    retrieved: List[Dict[str, Any]]
    reply: str
    next_action: Dict[str, Any]

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    intent = classify_intent(req.message)
    sentiment = sentiment_hint(req.message)

    docs = retrieve(req.message, k=3)

    system = "You are a customer support assistant. Be accurate and concise."
    user_prompt = build_grounded_prompt(req.message, docs)

    reply = llm.generate(system=system, user=user_prompt)

    action = next_best_action(intent, sentiment)

    # TODO: log to database / event stream (Kafka/PubSub/EventHub)
    return ChatResponse(
        intent=intent,
        sentiment=sentiment,
        retrieved=docs,
        reply=reply,
        next_action=action
    )

# Run: uvicorn app:app --reload --port 8000
