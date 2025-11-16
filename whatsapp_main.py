# whatsapp_main.py
import os
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from agente import app as agente_graph  # tu grafo original
import random

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")   # tu token EAAS...
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "glynne_verify_token")
PHONE_NUMBER_ID = "806755275863395"  # tu identificador de número

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 1) VERIFICACIÓN DEL WEBHOOK (Meta lo exige)
# ============================================================
@app.get("/webhook")
async def verify(request: Request):
    params = dict(request.query_params)

    if (
        params.get("hub.mode") == "subscribe"
        and params.get("hub.verify_token") == VERIFY_TOKEN
    ):
        return int(params["hub.challenge"])

    return {"error": "Verificación fallida"}

# ============================================================
# 2) RECIBIR MENSAJES DE WHATSAPP
# ============================================================
@app.post("/webhook")
async def webhook_handler(request: Request):
    body = await request.json()

    try:
        entry = body["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]

        if "messages" in value:
            message = value["messages"][0]
            sender = message["from"]
            text = message.get("text", {}).get("body", "")
            print(f"📩 Mensaje recibido: {text} de {sender}")

            # -----------------------------------------
            # Procesar con tu agente IA
            # -----------------------------------------
            estado = {
                "mensaje": text,
                "rol": "auditor",
                "user_id": sender,
                "historial": "",
                "respuesta": "",
            }

            result = agente_graph.invoke(estado)
            respuesta = result["respuesta"]

            # -----------------------------------------
            # Enviar respuesta al usuario por WhatsApp
            # -----------------------------------------
            send_whatsapp_message(sender, respuesta)

    except Exception as e:
        print("Error procesando mensaje:", e)

    return {"status": "ok"}

# ============================================================
# 3) FUNCIÓN PARA RESPONDER A WHATSAPP
# ============================================================
def send_whatsapp_message(to, message):

    url = f"https://graph.facebook.com/v22.0/{PHONE_NUMBER_ID}/messages"

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message}
    }

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, json=payload, headers=headers)

    print("📤 Respuesta enviada:", response.json())

    return response.json()

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("whatsapp_main:app", host="0.0.0.0", port=8000, reload=True)
