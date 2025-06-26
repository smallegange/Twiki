import streamlit as st
import json
import os
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import pandas as pd

st.title("Beheerpagina")

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(url="http://localhost:6333")

client = get_qdrant_client()
collections = client.get_collections().collections

def is_collection_public(collection_name):
    try:
        hits = client.scroll(
            collection_name=collection_name,
            limit=1,
            query_filter=Filter(must=[FieldCondition(key="visibility", match=MatchValue(value="public"))])
        )
        return bool(hits[0]) if hits else False
    except Exception:
        return False

if not collections:
    st.info("Geen collecties gevonden.")

# --- Configuratiebeheer ---
st.subheader("Configuratie: AI modellen")

config_path = "config/config.json"
if not os.path.exists(config_path):
    st.error("⚠️ config.json ontbreekt.")
else:
    with open(config_path, "r") as f:
        config = json.load(f)

    try:
        model_data = ollama.list()["models"]
        all_model_names = sorted([m["model"] for m in model_data])
    except Exception as e:
        all_model_names = []
        st.error(f"❌ Kan modellen niet ophalen via Ollama: {e}")

    #Beschikbare taalmodellen
    st.markdown("### Beschikbare taalmodellen (Agent)")
    selected_models = st.multiselect(
        "Selecteer taalmodellen",
        options=all_model_names,
        default=config.get("available_models", [])
    )
    config["available_models"] = selected_models

    #Reranking modellen
    st.markdown("### Reranking modellen")
    selected_rerank_models = st.multiselect(
        "Selecteer modellen voor reranking",
        options=all_model_names,
        default=config.get("rerank_models", [])
    )
    config["rerank_models"] = selected_rerank_models

    default_rerank_model = st.selectbox(
        "Standaard reranking model",
        config["rerank_models"],
        index=config["rerank_models"].index(config.get("default_rerank_model", config["rerank_models"][0])) if config.get("rerank_models") else 0
    )
    config["default_rerank_model"] = default_rerank_model

    #Embedding model
    st.markdown("### Embedding-model")
    embedding_options = ["-- Geen embedding-model --"] + all_model_names
    embedding_default = config.get("embedding_model", embedding_options[0])
    embedding_model = st.selectbox("Selecteer embedding-model", options=embedding_options, index=embedding_options.index(embedding_default) if embedding_default in embedding_options else 0)
    config["embedding_model"] = embedding_model if embedding_model != "-- Geen embedding-model --" else None

    st.markdown("### GPT-4 toegestaan bij collecties")
    available_collections = [c.name for c in collections]
    selected_collections = st.multiselect(
        "Selecteer collecties waarvoor GPT-4 beschikbaar is",
        options=available_collections,
        default=[c for c in config.get("allowed_collections_for_gpt4", []) if c in available_collections]
    )
    config["allowed_collections_for_gpt4"] = selected_collections

    if st.button("Config opslaan"):
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        st.success("Configuratie opgeslagen.")

# --- API-sleutels beheren (.env in hoofdmap) ---
st.subheader("API-sleutel instellen")

# Vind pad naar hoofdmap (één map omhoog vanaf /pages)
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
env_path = os.path.join(main_dir, ".env")
env_vars = {}

# Bestaande env inladen (indien aanwezig)
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "=" in line:
                key, val = line.strip().split("=", 1)
                env_vars[key] = val

# Toon huidig opgeslagen sleutel (masked)
api_key = st.text_input("OpenAI API-key", env_vars.get("OPENAI_API_KEY", ""), type="password")

# Opslaan
if st.button("Sla API-sleutel op"):
    env_vars["OPENAI_API_KEY"] = api_key
    try:
        with open(env_path, "w") as f:
            for key, val in env_vars.items():
                f.write(f"{key}={val}\n")
        st.success(".env bestand in hoofdmap succesvol opgeslagen.")
    except Exception as e:
        st.error(f"Fout bij opslaan: {e}")

# Test API key knop
if st.button("Test API-key"):
    import openai
    try:
        client = openai.OpenAI(api_key=api_key)
        # Probeer een eenvoudige API call (lijst modellen)
        client.models.list()
        st.success("✅ Verbinding met OpenAI API is succesvol!")
    except Exception as e:
        st.error(f"❌ Kan geen verbinding maken met OpenAI API: {e}")

