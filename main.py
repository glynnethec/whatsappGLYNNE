# ============================
# GLYNNE AI - API + WhatsApp (FINAL CORREGIDO)
# ============================

import os
import random
import requests
from dotenv import load_dotenv
from typing import TypedDict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse  # ✅ CORRECCIÓN 1
from pydantic import BaseModel

# LangChain / LangGraph
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# ============================
# 1) VARIABLES
# ============================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "glynne_verify")  # ESTE TOKEN VA EN META
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")

# ============================
# 2) LLM GROQ
# ============================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.4,
)

# ============================
# 3) PROMPT
# ============================
Prompt_estructura = """
[META]
Tu objetivo es actuar como un experto en GLYNNE y explicar todos los servicios, soluciones y proyectos que ofrece la empresa. Analiza la situación del usuario y relaciona cada recomendación con cómo la inteligencia artificial y la arquitectura de software de GLYNNE pueden potenciar el crecimiento y la eficiencia empresarial.

[GUÍA DE SERVICIOS DE GLYNNE]
1. **Automatización de procesos empresariales**: Optimización de flujos de trabajo internos y externos mediante software personalizado y agentes IA.
2. **Desarrollo de software a medida**: Creación de sistemas y plataformas adaptadas a las necesidades específicas de cada empresa.
3. **Integración de inteligencia artificial**: Implementación de LLMs, agentes conversacionales y sistemas predictivos para mejorar la toma de decisiones.
4. **Auditorías de procesos con IA**: Diagnóstico empresarial automatizado que identifica cuellos de botella y oportunidades de eficiencia.
5. **CRM multicanal y automatización de ventas**: Gestión centralizada de clientes mediante WhatsApp, Gmail y otros canales, con flujos automatizados.
6. **Generación de propuestas técnicas y consultoría estratégica**: Transformación de auditorías y análisis en planes accionables para la empresa.
7. **Arquitectura empresarial escalable**: Diseño de sistemas acoplables que soportan crecimiento, integración de IA y conectividad con APIs externas.

[FORMATO DE RESPUESTA]
- Explica los servicios relacionados con la consulta del usuario.
- Profesional, corporativa, clara y concisa.
- Máximo 100 palabras.
- Orientada a visión empresarial y estratégica, usando ejemplos prácticos de GLYNNE.

[ADVERTENCIAS]
- No saludes ni uses frases informales.
- No inventes información ni supuestos.
- Mantén un tono corporativo, directo y accionable.

[MEMORIA]
Usa siempre el contexto de la memoria: {historial}, incluyendo proyectos, servicios, herramientas implementadas y aprendizajes previos de GLYNNE.

[ENTRADA DEL USUARIO]
consulta: {mensaje}

respuesta:
"""

prompt = PromptTemplate(
    input_variables=["rol", "mensaje", "historial"],
    template=Prompt_estructura.strip(),
)

# ============================
# 4) MEMORIA POR USUARIO
# ============================
usuarios = {}

def get_memory(user_id: str):
    if user_id not in usuarios:
        usuarios[user_id] = ConversationBufferMemory(
            memory_key="historial",
            input_key="mensaje"
        )
    return usuarios[user_id]

# ============================
# 5) LANGGRAPH
# ============================
class State(TypedDict):
    mensaje: str
    rol: str
    historial: str
    respuesta: str
    user_id: str

def agente_node(state: State) -> State:
    memory = get_memory(state["user_id"])
    historial = memory.load_memory_variables({}).get("historial", "")

    final_prompt = prompt.format(
        rol=state["rol"],
        mensaje=state["mensaje"],
        historial=historial
    )

    respuesta = llm.invoke(final_prompt).content

    memory.save_context({"mensaje": state["mensaje"]}, {"respuesta": respuesta})

    state["respuesta"] = respuesta
    state["historial"] = historial
    return state

workflow = StateGraph(State)
workflow.add_node("agente", agente_node)
workflow.set_entry_point("agente")
workflow.add_edge("agente", END)

agente_graph = workflow.compile()

# ============================
# 6) FASTAPI
# ============================
app = FastAPI(title="GLYNNE AI – Unified Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# 7) API /chat
# ============================
class ChatInput(BaseModel):
    mensaje: str
    rol: str = "auditor"
    user_id: str | None = None

class ChatOutput(BaseModel):
    user_id: str
    respuesta: str

@app.post("/chat")
async def chat_endpoint(data: ChatInput):

    user_id = data.user_id or str(random.randint(10000, 99999))

    estado = {
        "mensaje": data.mensaje,
        "rol": data.rol,
        "user_id": user_id,
        "historial": "",
        "respuesta": "",
    }

    result = agente_graph.invoke(estado)

    return ChatOutput(
        user_id=user_id,
        respuesta=result["respuesta"]
    )

# ============================
# 8) META WEBHOOK (GET) - ✅ CORRECCIÓN 1
# ============================
@app.get("/webhook", response_class=PlainTextResponse)  # Meta exige texto plano
async def verify(request: Request):
    params = dict(request.query_params)

    if (
        params.get("hub.mode") == "subscribe"
        and params.get("hub.verify_token") == VERIFY_TOKEN
    ):
        return params.get("hub.challenge")  # Devuelve solo el string

    return {"error": "token incorrecto"}

# ============================
# 9) RECIBIR MENSAJES WHATSAPP
# ============================
@app.post("/webhook")
async def webhook_handler(request: Request):
    body = await request.json()

    try:
        entry = body["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]

        if "messages" in value:
            msg = value["messages"][0]
            sender = msg["from"]
            text = msg.get("text", {}).get("body", "")

            estado = {
                "mensaje": text,
                "rol": "auditor",
                "user_id": sender,
                "historial": "",
                "respuesta": "",
            }

            result = agente_graph.invoke(estado)
            respuesta = result["respuesta"]

            send_whatsapp_message(sender, respuesta)

    except Exception as e:
        print("❌ Error:", e)

    return {"ok": True}

# ============================
# 10) FUNCIÓN PARA RESPONDER - ✅ CORRECCIÓN 2
# ============================
def send_whatsapp_message(to, message):

    # ✅ CORREGIDO: Eliminado espacio extra antes de {PHONE_NUMBER_ID}
    url = f"https://graph.facebook.com/v22.0/{PHONE_NUMBER_ID}/messages"

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message},
    }

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    r = requests.post(url, json=payload, headers=headers)
    print("📤 Enviado:", r.json())
    return r.json()


# ============================
# LOCAL RUN
# ============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)