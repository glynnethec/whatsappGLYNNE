# main.py
import random
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Importar TODO desde tu componente original sin modificarlo
from agent.chat import app as agente_graph, roles, get_memory

# ========================================================
# FastAPI
# ========================================================
app_api = FastAPI(title="GLYNNE AI - API")

# CORS para que el front en Next.js pueda consumirlo
app_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================================
# Modelos
# ========================================================
class ChatInput(BaseModel):
    mensaje: str
    rol: str = "auditor"
    user_id: str | None = None


class ChatOutput(BaseModel):
    user_id: str
    respuesta: str


# ========================================================
# Endpoint principal
# ========================================================
@app_api.post("/chat", response_model=ChatOutput)
async def chat(data: ChatInput):

    # user_id dinámico si no lo envían
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
        respuesta=result["respuesta"],
    )


# ========================================================
# Para correrlo con: uvicorn main:app_api --reload
# ========================================================
if __name__ == "__main__":
    import uvicorn
    print("🚀 API lista en http://localhost:8000/chat")
    uvicorn.run("main:app_api", host="0.0.0.0", port=8000, reload=True)
