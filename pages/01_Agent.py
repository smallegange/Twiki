import json

import streamlit as st
import pandas as pd
import numpy as np
import ast
import time
import os
import csv
import ollama
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv, dotenv_values
from openai import OpenAI
import io
import re

# Laadt OpenAI key from .env (altijd vers ophalen uit hoofdmap)
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
env_path = os.path.join(main_dir, ".env")
openai_api_key = None
if os.path.exists(env_path):
    env_vars = dotenv_values(env_path)
    openai_api_key = env_vars.get("OPENAI_API_KEY")
else:
    openai_api_key = os.getenv("OPENAI_API_KEY")

openai_client = OpenAI(api_key=openai_api_key)

st.set_page_config(page_title="Twiki Serve", layout="wide")


st.markdown("<h1 style='margin-top: 15px;'> Hallo! Hoe kan ik helpen?</h1>", unsafe_allow_html=True)

@st.cache_resource
def get_qdrant_client():
    # Gebruik de server-URL, niet het lokale path!
    return QdrantClient(url="http://localhost:6333")
    qdrant_path = os.path.abspath(os.path.join(base_dir, "..", "qdrant_data", "collections"))
    return QdrantClient(path=qdrant_path)

client = get_qdrant_client()
collections = client.get_collections().collections
collection_names = [c.name for c in collections] if collections else []

if not collection_names:
    st.warning("Geen collecties gevonden in Qdrant.")
    st.stop()

selected_collections = st.sidebar.multiselect(
    "Kies één of meer collecties", collection_names, default=[collection_names[0]]
)


with open("config/config.json") as f:
    config = json.load(f)

available_models = config["available_models"]
allowed_collections = set(config["allowed_collections_for_gpt4"])

if set(selected_collections) <= allowed_collections:
    available_models.append("openai:gpt-4-turbo")

#Filters en prompt nog niet geïmplementeerd in de config.json, maar wel in de code. Nog opschonen.
model_choice = st.sidebar.selectbox("Kies taalmodel", available_models)
k_limit = st.sidebar.slider("Aantal top resultaten", 100, 1500, 600, step=100)
debug_mode = st.sidebar.checkbox("Debug mode")

with st.sidebar.expander("Geavanceerde instellingen", expanded=False):
    rerank_model = st.selectbox("Reranking model", config.get("rerank_models", []), index=config.get("rerank_models", []).index(config.get("default_rerank_model", "")))
    use_reranking = st.checkbox("Gebruik reranking", value=False)
    top_n_chunks = st.slider("Aantal fragmenten na reranking", 1, 60, 10)
    top_n_no_rerank = st.slider("Aantal fragmenten zonder reranking", 1, 100, 100)
    temperature = st.slider("Temperatuur", 0.0, 1.0, 0.25, step=0.05)
    top_p = st.slider("Top-p", 0.0, 1.0, 0.95, step=0.05)
    repeat_penalty = st.slider("Repeat penalty", 0.5, 2.0, 1.1, step=0.1)
    log_performance = st.checkbox("Log performance naar CSV", value=False)
    export_path = st.text_input("Pad naar logbestand (CSV)", value="./performance/performance_log.csv")

if model_choice == "openai:gpt-4-turbo" and not set(selected_collections) <= allowed_collections:
    st.sidebar.warning(f"GPT-4 is alleen beschikbaar met collecties: {sorted(allowed_collections)}")
    st.stop()

if st.sidebar.button("Sessie afsluiten"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.clear()
    st.success("Sessie is beëindigd. Je kunt het venster sluiten.")
    st.stop()

embedding_model = OllamaEmbeddings(model=config.get("embedding_model", "nomic-embed-text"))
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Stel je vraag...")

def generate_query_embedding(text):
    return embedding_model.embed_query(text)

def build_rerank_prompt(vraag, chunks, top_n_chunks, max_chunks=40):
    context = "\n".join([f"{i + 1}. {chunk}" for i, chunk in enumerate(chunks[:max_chunks])])
    return (
        f"Je krijgt een vraag en een lijst met tekstfragmenten (genummerd).\n"
        f"Kies de {top_n_chunks} meest relevante fragmenten (geef alleen de nummers).\n\n"
        f"Vraag: {vraag}\n\nFragmenten:\n{context}\n\n"
        f"Antwoord als een Python-lijst, bv: [1, 4, 5]"
    )

def build_query_filter(user_input):
    maand_map = {
        "januari": "januari", "februari": "februari", "maart": "maart", "april": "april",
        "mei": "mei", "juni": "juni", "juli": "juli", "augustus": "augustus",
        "september": "september", "oktober": "oktober", "november": "november", "december": "december"
    }
    must_filters = []
    lower_q = user_input.lower()

    for maand in maand_map:
        if maand in lower_q:
            must_filters.append(FieldCondition(key="maand_naam", match=MatchValue(value=maand)))
            break

    if match := re.search(r"medewerker\s*(\d{3,6})", lower_q):
        must_filters.append(FieldCondition(key="medewerker_id", match=MatchValue(value=match.group(1))))

    if match := re.search(r"(dienst|shift)[^\d]*(\d{1,6})", lower_q):
        must_filters.append(FieldCondition(key="dienst_id", match=MatchValue(value=match.group(2))))

    if any(term in lower_q for term in ["verlof", "vakantie", "vrij", "vrije dag", "afwezig"]):
        must_filters.append(FieldCondition(key="verlof_conflict", match=MatchValue(value=True)))
    if "niet beschikbaar" in lower_q or "onbeschikbaar" in lower_q:
        must_filters.append(FieldCondition(key="beschikbaarheid_conflict", match=MatchValue(value=True)))
    if any(term in lower_q for term in ["overdracht", "overgedragen", "geruild", "gewisseld"]):
        must_filters.append(FieldCondition(key="overdracht", match=MatchValue(value=True)))

    return Filter(must=must_filters) if must_filters else None


if user_input and user_input.strip():
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    query_vector = generate_query_embedding(user_input)
    query_filter = build_query_filter(user_input)

    if debug_mode:
        st.write("Actieve filter:", query_filter.dict() if query_filter else "Geen")

    all_results = []
    for collection in selected_collections:
        results = client.search(collection_name=collection, query_vector=query_vector, limit=k_limit, query_filter=query_filter)
        all_results.extend(results)

    top_k_chunks = [hit.payload.get("content", "") for hit in all_results if "content" in hit.payload]
    if not top_k_chunks:
        st.warning("Geen resultaten gevonden.")
        st.stop()

    reranked_chunks = top_k_chunks[:top_n_no_rerank]
    rerank_duration = 0.0

    if use_reranking:
        rerank_prompt = build_rerank_prompt(user_input, top_k_chunks, top_n_chunks)
        if debug_mode:
            with st.expander("🔍 Reranking prompt"):
                st.code(rerank_prompt[:20000])
        rerank_start = time.time()
        rerank_response = ollama.chat(model=rerank_model, messages=[{"role": "user", "content": rerank_prompt}])
        rerank_duration = round(time.time() - rerank_start, 2)
        selected_indices = ast.literal_eval(rerank_response['message']['content'])
        reranked_chunks = [top_k_chunks[i - 1] for i in selected_indices if 1 <= i <= len(top_k_chunks)]
        if debug_mode:
            st.info(f"Reranking model: {rerank_model}")
            st.info(f"Geselecteerde fragmenten: {selected_indices}")

    context = "\n".join([f"- {chunk}" for chunk in reranked_chunks])
    if model_choice == "openai:gpt-4-turbo":
        prompt = f"Je bent een AI-expert en adviseur en beantwoordt de vraag in het Nederlands en primair met behulp van de volgende data:\n\n{context}\n\nVraag: {user_input}\n\nAls het antwoord niet in de data staat, zeg: 'Niet gevonden in de data.'"
    else:
        prompt = f"Beantwoord de volgende vraag in het Nederlands en uitsluitend met behulp van deze data:\n\n{context}\n\nVraag: {user_input}\n\nAls het antwoord niet in de data staat, zeg: 'Niet gevonden in de data.'"

    if debug_mode:
        with st.expander("Prompt aan LLM"):
            st.code(prompt[:20000])

    start_time = time.time()
    if model_choice == "openai:gpt-4-turbo":
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                presence_penalty=repeat_penalty,
            )
            response_text = response.choices[0].message.content
            total_tokens = response.usage.total_tokens
        except Exception as e:
            st.error(f"❌ Kan geen verbinding maken met OpenAI API: {e}")
            st.stop()
    else:
        response = ollama.chat(
            model=model_choice,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature, "top_p": top_p, "repeat_penalty": repeat_penalty}
        )
        response_text = response['message']['content']
        total_tokens = response.get('eval_count', None)

    duration = round(time.time() - start_time, 2)
    st.session_state.chat_history.append({"role": "assistant", "content": response_text})

    # --- Info bar: tijd, model, rerank, tokens (alleen voor OpenAI) ---
    info_html = f"<div style='font-size: 0.8em; color: gray; margin-top: 5px;'>Beantwoord in {duration} seconden met model <code>{model_choice}</code>. Reranking duurde {rerank_duration} seconden met model <code>{rerank_model}</code>." 
    if model_choice == "openai:gpt-4-turbo" and total_tokens:
        info_html += f" Tokens: <code>{total_tokens}</code>."
    info_html += "</div>"

    with st.chat_message("assistant"):
        st.markdown(response_text)
        st.markdown(info_html, unsafe_allow_html=True)

# # Dall E integratie nog niet geïmplementeerd. Komt uit specifieke versie.
if "robin" in selected_collections:
    st.header("Genereer visualisatie")
    if "dalle_image_url" not in st.session_state:
        st.session_state.dalle_image_url = None

    custom_prompt = st.text_area(
        "Geef een creatieve prompt voor DALL·E",
        placeholder="Bijv: Teken een kleurrijke roadmap in de vorm van een bordspel over ...'s groei op vier domeinen...",
        key="dalle_custom_prompt"
    )
    generate_button = st.button("Genereer afbeelding")

    if generate_button and not custom_prompt.strip():
        st.warning("⚠️ Geef eerst een prompt in voordat je een afbeelding genereert.")

    if generate_button and custom_prompt.strip():
        last_user_prompt = st.session_state.chat_history[-2]["content"] if len(st.session_state.chat_history) >= 2 else ""
        last_assistant_response = st.session_state.chat_history[-1]["content"] if len(st.session_state.chat_history) >= 1 else ""
        dalle_prompt = (
            "Je bent een cliëntbegeleider en een creatieve geest. "
            f"Gebruik deze context:\n\nVraag: {last_user_prompt}\n\nAntwoord: {last_assistant_response}\n\n"
            f"Opdracht: {custom_prompt.strip()}"
        )
        with st.spinner("Afbeelding wordt gegenereerd..."):
            try:
                dalle_response = openai_client.images.generate(
                    model="dall-e-3",
                    prompt=dalle_prompt,
                    size="1024x1024",
                    n=1
                )
                image_url = dalle_response.data[0].url
                st.session_state.dalle_image_url = image_url
            except Exception as e:
                st.error(f"❌ Fout bij genereren afbeelding: {e}")

    if st.session_state.dalle_image_url:
        st.image(st.session_state.dalle_image_url, caption="DALL·E visualisatie", use_container_width=True)
        try:
            image_bytes = requests.get(st.session_state.dalle_image_url).content
            st.download_button(
                label="📥 Download afbeelding",
                data=image_bytes,
                file_name="robin_roadmap.png",
                mime="image/png"
            )
        except Exception as e:
            st.error(f"❌ Fout bij downloaden afbeelding: {e}")

# --- Q&A export functionaliteit ---
qa_pairs = []
history = st.session_state.get("chat_history", [])
for i in range(0, len(history) - 1, 2):
    if (
        "role" in history[i] and history[i]["role"] == "user"
        and "role" in history[i + 1] and history[i + 1]["role"] == "assistant"
    ):
        question = history[i].get("content", "").replace("\n", " ").replace(",", " ")
        answer = history[i + 1].get("content", "").replace("\n", " ").replace(",", " ")
        qa_pairs.append((question, answer))

if qa_pairs:
    csv_data = "Vraag,Antwoord\n" + "\n".join(f"{q},{a}" for q, a in qa_pairs)
    csv_buffer = io.StringIO(csv_data)
    st.download_button(
        label="Exporteer Q&A",
        data=csv_buffer.getvalue(),
        file_name="chat_export.csv",
        mime="text/csv"
    )

# --- Performance log aanmaken indien niet aanwezig ---
performance_dir = "./performance"
performance_file = os.path.join(performance_dir, "performance_log.csv")

if not os.path.exists(performance_dir):
    os.makedirs(performance_dir, exist_ok=True)

if not os.path.exists(performance_file):
    with open(performance_file, "w", encoding="utf-8") as f:
        f.write("timestamp,model,rerank_model,duration,rerank_duration,total_tokens,prompt_tokens,completion_tokens\n")

# --- Spraak naar tekst (Whisper) upload & transcriptie ---
st.sidebar.markdown("---")
st.sidebar.markdown("**Spraak naar tekst:**")
# Status label voor spraaksectie
whisper_mode = st.sidebar.radio(
    "Kies transcriptie-methode:",
    options=["Online (OpenAI)", "Offline (Lokaal)"],
    index=0,
    key="whisper_mode_radio_sidebar"
)

audio_file = st.sidebar.file_uploader(
    "Upload audiobestand (mp3, wav, m4a)",
    type=["mp3", "wav", "m4a"],
    key="audio_uploader"
)
transcribed_text = None
if audio_file is not None:
    if st.sidebar.button("Transcribeer audio"):
        with st.spinner("Transcriberen..."):
            try:
                if whisper_mode == "Online (OpenAI)":
                    transcript = openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text",
                        language="nl"
                    )
                    transcribed_text = transcript
                else:
                    import tempfile
                    import whisper
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                        tmp_audio.write(audio_file.read())
                        tmp_audio.flush()
                        model = whisper.load_model("base")
                        result = model.transcribe(tmp_audio.name, language="nl")
                        transcribed_text = result["text"]
                st.sidebar.success("Transcriptie voltooid!")
            except Exception as e:
                st.sidebar.error(f"Fout bij transcriptie: {e}")
    if transcribed_text:
        st.sidebar.markdown("**Transcriptie:**")
        st.sidebar.write(transcribed_text)
        if st.sidebar.button("Gebruik als vraag"):
            st.session_state["chat_input"] = transcribed_text
            
if whisper_mode == "Online (OpenAI)":
    st.sidebar.markdown("<div style='background: #e0ffe0; color: #217a2b; border-radius: 8px; padding: 4px 12px; display: inline-block; font-weight: 600; margin-bottom: 8px;'>🟢 Online</div>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("<div style='background: #f5f5f5; color: #444; border-radius: 8px; padding: 4px 12px; display: inline-block; font-weight: 600; margin-bottom: 8px;'>⚡ Lokaal</div>", unsafe_allow_html=True)