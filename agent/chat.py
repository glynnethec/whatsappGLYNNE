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
    raise ValueError("en el .env  no hay una api valida")

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

