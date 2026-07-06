"""
Streamlit Frontend — Wellness and Health Promotion Chatbot
Exercise 3 — Phase 4 (ECBTI)

Requires the following files to be in the same folder (exported from the Exercise 2 notebook):
  - config.yaml
  - kb_faiss_index/  (FAISS vector index)

Run locally:
  streamlit run app.py

Deploy for free on:
  Streamlit Community Cloud or Hugging Face Spaces (see DEPLOY.md)
  
"""

import os
import time
import json
import csv
import re
import textwrap
import unicodedata
from datetime import datetime, timezone

import yaml
import streamlit as st


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

@st.cache_resource
def load_config():
    if not os.path.exists(CONFIG_PATH):
        st.error(
            "No se encontró config.yaml. Copia aquí el archivo generado por el notebook "
            "del Ejercicio 2 (backend)."
        )
        st.stop()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CONFIG = load_config()


CRISIS_PATTERNS = [
    r"\bquiero morir\b",
    r"\bquitarme la vida\b",
    r"\bsuicidarme\b",
    r"\bhacerme da[nñ]o\b",
    r"\bno quiero vivir\b",
    r"\bdolor (agudo|muy fuerte) en el pecho\b",
    r"\bsobredosis\b",
]

ESCALATION_MESSAGE = (
    "Lamento que estés pasando por esto. No estoy en capacidad de ayudarte con una crisis "
    "o emergencia: por favor contacta de inmediato a una línea de ayuda profesional o a los "
    "servicios de urgencias de tu país. En Colombia puedes llamar a la Línea 123 (emergencias) "
    "o a la Línea de Salud Mental 106. Si tienes a alguien de confianza cerca, avísale ahora."
)

SYSTEM_PROMPT = textwrap.dedent("""
Eres un asistente virtual de bienestar y promoción de la salud, dirigido a estudiantes y
público general. Tu conocimiento se limita a hábitos saludables generales: sueño, actividad
física, nutrición básica y manejo del estrés, basado en fuentes públicas como la OMS y el CDC.

Reglas obligatorias:
1. NUNCA emitas diagnósticos médicos ni interpretes síntomas específicos de una persona.
2. NUNCA recomiendes medicamentos, dosis ni tratamientos.
3. Si el CONTEXTO RECUPERADO no contiene información suficiente, dilo explícitamente.
4. Si la consulta es ambigua, pide una aclaración concreta antes de responder.
5. Responde siempre en español, en tono cálido, claro y breve (máximo 4-6 frases).
6. Recuerda que la información es educativa y no reemplaza a un profesional de la salud.
""").strip()

PROMPT_TEMPLATE = textwrap.dedent("""
{system_prompt}

### Contexto recuperado (base de conocimiento):
{context}

### Historial reciente de la conversación:
{history}

### Consulta actual del usuario:
{query}

### Respuesta del asistente:
""").strip()


def clean_input(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch == "\n")
    return text


def detect_crisis(text: str) -> bool:
    text_norm = text.lower()
    return any(re.search(pat, text_norm) for pat in CRISIS_PATTERNS)


def guess_topic(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ["dormir", "sueno", "sueño", "insomnio", "descansar"]):
        return "sueno"
    if any(k in q for k in ["ejercicio", "actividad fisica", "actividad física", "entrenar", "correr"]):
        return "actividad_fisica"
    if any(k in q for k in ["comer", "dieta", "nutricion", "nutrición", "alimentacion", "alimentación"]):
        return "alimentacion_saludable"
    if any(k in q for k in ["estres", "estrés", "ansiedad", "animo", "ánimo", "mental"]):
        return "salud_mental"
    return "general"


def build_prompt(query: str, context: str, history: str) -> str:
    return PROMPT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        context=context if context.strip() else "(sin resultados relevantes)",
        history=history if history.strip() else "(sin turnos previos)",
        query=query,
    )



@st.cache_resource(show_spinner="Cargando modelo de embeddings...")
def get_embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=CONFIG["embeddings"]["model_name"])


@st.cache_resource(show_spinner="Cargando base de conocimiento (FAISS)...")
def get_vectorstore():
    from langchain_community.vectorstores import FAISS
    path = os.path.join(BASE_DIR, CONFIG["rag"]["vectorstore_path"])
    if not os.path.exists(path):
        st.warning(
            f"No se encontró el índice vectorial en '{path}/'. El chatbot funcionará sin "
            "recuperación de contexto (RAG) hasta que lo agregues."
        )
        return None
    embedding_model = get_embedding_model()
    return FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)


@st.cache_resource(show_spinner="Cargando el modelo de lenguaje (puede tardar unos minutos)...")
def get_llm():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
    from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint

    provider = CONFIG["llm"]["provider"]
    if provider == "local":
        model_name = CONFIG["llm"]["model_name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        gen_pipeline = hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=CONFIG["llm"]["max_new_tokens"],
            temperature=CONFIG["llm"]["temperature"],
            top_p=CONFIG["llm"]["top_p"],
            do_sample=True,
            return_full_text=False,
        )
        return HuggingFacePipeline(pipeline=gen_pipeline)
    elif provider == "hf_api":
        from huggingface_hub import InferenceClient

        class RemoteLLM:
            """Adaptador mínimo: expone .invoke(prompt) usando la vía moderna
            de Inference Providers (chat completions)."""
            def __init__(self, model, temperature, max_tokens):
                self.model = model
                self.temperature = temperature
                self.max_tokens = max_tokens
                self.client = InferenceClient(
                    token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
                )

            def invoke(self, prompt: str) -> str:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return resp.choices[0].message.content

        return RemoteLLM(
            CONFIG["llm"]["hf_api_model"],
            CONFIG["llm"]["temperature"],
            CONFIG["llm"]["max_new_tokens"],
        )
    raise ValueError(f"Proveedor de LLM no soportado: {provider}")


def retrieve_context(query: str) -> str:
    vs = get_vectorstore()
    if vs is None:
        return ""
    results = vs.similarity_search(query, k=CONFIG["rag"]["top_k"])
    parts = [f"[Fuente: {r.metadata.get('topic', 'desconocido')}] {r.page_content}" for r in results]
    return "\n\n".join(parts)



LOG_PATH = CONFIG["logging"]["log_path"]
LOG_FIELDS = ["timestamp", "session_id", "topic", "query", "response",
              "escalated", "context_chars", "response_time_s"]


def log_interaction(session_id, topic, query, response, escalated, context_chars, response_time_s):
    file_exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "topic": topic,
            "query": query,
            "response": response,
            "escalated": escalated,
            "context_chars": context_chars,
            "response_time_s": round(response_time_s, 2),
        })



if "messages" not in st.session_state:
    st.session_state.messages = []  # para mostrar en la UI: [{"role": "user"/"assistant", "content": str}]
if "turns" not in st.session_state:
    st.session_state.turns = []  # para el prompt: [(user, bot), ...]
if "session_id" not in st.session_state:
    st.session_state.session_id = f"web-{int(time.time())}"


def memory_as_text() -> str:
    window = CONFIG["memory"]["window_size"]
    max_chars = CONFIG["memory"]["max_context_chars"]
    recent = st.session_state.turns[-window:]
    lines = []
    for u, b in recent:
        lines.append(f"Usuario: {u}")
        lines.append(f"Asistente: {b}")
    text = "\n".join(lines)
    return text[-max_chars:] if len(text) > max_chars else text



st.set_page_config(page_title="Asistente de Bienestar", page_icon="🌱", layout="centered")

st.title("🌱 Asistente de Bienestar y Promoción de la Salud")
st.caption(
    "Chatbot educativo basado en un LLM open-source con RAG · Fase 4 ECBTI · "
    "No reemplaza la consulta con un profesional de la salud."
)

with st.expander("ℹ️ Qué puede y qué NO puede hacer este chatbot", expanded=False):
    st.markdown("""
**Puede ayudarte con:**
- Información general sobre sueño, actividad física, alimentación y manejo del estrés.
- Recomendaciones basadas en fuentes públicas (OMS, CDC).

**No puede:**
- Diagnosticar enfermedades ni interpretar síntomas específicos.
- Recomendar medicamentos o dosis.
- Reemplazar la atención de un profesional de la salud.

**Prompts recomendados:**
- "¿Cuántos minutos de ejercicio a la semana recomienda la OMS?"
- "Dame 3 consejos para dormir mejor."
- "¿Qué es una dieta equilibrada según la OMS?"

**Limitaciones conocidas:** el modelo puede alucinar o dar respuestas incompletas ante temas
fuera de su base de conocimiento; siempre verifica la información importante con fuentes
oficiales o un profesional.
""")

with st.sidebar:
    st.header("⚙️ Configuración")
    st.write(f"**Modelo:** `{CONFIG['llm']['model_name'] if CONFIG['llm']['provider']=='local' else CONFIG['llm']['hf_api_model']}`")
    temperature = st.slider("Temperatura", 0.0, 1.0, float(CONFIG["llm"]["temperature"]), 0.05)
    top_k = st.slider("Top-k de recuperación (RAG)", 1, 8, int(CONFIG["rag"]["top_k"]))
    max_tokens = st.slider("Máx. tokens de respuesta", 50, 500, int(CONFIG["llm"]["max_new_tokens"]), 10)
    CONFIG["llm"]["temperature"] = temperature
    CONFIG["rag"]["top_k"] = top_k
    CONFIG["llm"]["max_new_tokens"] = max_tokens

    st.divider()
    if st.button("🗑️ Reiniciar conversación"):
        st.session_state.messages = []
        st.session_state.turns = []
        st.rerun()

# Historial visual
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrada del usuario (multi-turno)
user_input = st.chat_input("Escribe tu pregunta sobre bienestar y salud...")

if user_input:
    query = clean_input(user_input)
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            t0 = time.time()

            if detect_crisis(query):
                response = ESCALATION_MESSAGE
                escalated, topic, ctx_len = True, "crisis", 0
            else:
                context = retrieve_context(query)
                history = memory_as_text()
                prompt = build_prompt(query, context, history)
                llm = get_llm()
                raw = llm.invoke(prompt)
                response = raw.strip() if isinstance(raw, str) else str(raw)
                escalated, topic, ctx_len = False, guess_topic(query), len(context)

            elapsed = time.time() - t0
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.turns.append((query, response))
    log_interaction(st.session_state.session_id, topic, query, response, escalated, ctx_len, elapsed)

st.divider()
st.caption(
    "⚠️ Este chatbot ofrece información educativa general y no constituye consejo médico. "
    "Ante una emergencia, contacta a los servicios de urgencias de tu país."
)
