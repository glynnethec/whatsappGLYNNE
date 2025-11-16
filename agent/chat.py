import os, random
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# ========================
# 1. Configuración
# ========================
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("en el .env no hay una api valida")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0.4,
)

# ========================
# 2. Prompt
# ========================
Prompt_estructura = """
[META]
tu meta es analizar el negocio del usuario, hacer consulta puntual
y poder generar un diagnóstico de cómo la IA puede mejorar el crecimiento empresarial,
explicando como si fueras un {rol} profesional. 

[Formato Respuesta]
La respuesta debe ser clara en base a {rol}, no más de 100 palabras, profesional y corporativa. 

[ADVERTENCIA]
- No saludes en cada consulta 
- No inventes datos 
- Mantén el tono conciso 

[MEMORIA]
Usa siempre el contexto de la memoria: {historial}

[ENTRADA DEL USUARIO]
consulta: {mensaje}

respuesta:
"""

prompt = PromptTemplate(
    input_variables=["rol", "mensaje", "historial"],
    template=Prompt_estructura.strip(),
)

# ========================
# 3. Estado global
# ========================
class State(TypedDict):
    mensaje: str
    rol: str
    historial: str
    respuesta: str
    user_id: str


# memoria por usuario
usuarios = {}

def get_memory(user_id: str):
    if user_id not in usuarios:
        usuarios[user_id] = ConversationBufferMemory(
            memory_key="historial", input_key="mensaje"
        )
    return usuarios[user_id]

# ========================
# 4. Nodo principal
# ========================
def agente_node(state: State) -> State:
    memory = get_memory(state.get("user_id", "default"))
    historial = memory.load_memory_variables({}).get("historial", "")

    texto_prompt = prompt.format(
        rol=state["rol"], mensaje=state["mensaje"], historial=historial
    )
    respuesta = llm.invoke(texto_prompt).content

    # guardar en memoria
    memory.save_context({"mensaje": state["mensaje"]}, {"respuesta": respuesta})

    # actualizar estado
    state["respuesta"] = respuesta
    state["historial"] = historial
    return state

# ========================
# 5. Construcción del grafo
# ========================
workflow = StateGraph(State)
workflow.add_node("agente", agente_node)
workflow.set_entry_point("agente")
workflow.add_edge("agente", END)
app = workflow.compile()

# ========================
# 6. CLI interactiva
# ========================
print("LLM iniciado con LangGraph")

roles = {
    "auditor": "actua como un auditor empresarial...",
    "desarrollador": "explica con detalle técnico...",
    "vendedor": "vende software con mala técnica...",
}

user_id = str(random.randint(10000, 90000))
print(f"tu user id es {user_id}")

rol = "auditor"

