import streamlit as st
import os
import sys
import shutil
import json
import hashlib
import time
from langchain_community.embeddings import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

UPLOAD_DIR = "./data/upload"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title("📄 Bestanden uploaden en embedden")

# Initialiseer sessiestatus
if "step" not in st.session_state:
    st.session_state.step = 1
if "embedding_running" not in st.session_state:
    st.session_state.embedding_running = False

# Stap 1: uploaden
if st.session_state.step == 1:
    uploaded_file = st.file_uploader(
        "Upload een bestand (PDF, CSV, TXT, DOCX, MD, JSON, XLSX)",
        type=["pdf", "csv", "txt", "docx", "md", "json", "xlsx"],
        key="file_uploader",
    )
    if uploaded_file is not None:
        save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Bestand opgeslagen: {uploaded_file.name}")

        if st.button("Ga verder"):
            st.session_state.uploaded_file = uploaded_file.name
            st.session_state.step = 2
            st.stop()
    else:
        st.info("Upload eerst een bestand om verder te gaan.")

# Stap 2: embedden
elif st.session_state.step == 2:
    uploaded_files = os.listdir(UPLOAD_DIR)
    if not uploaded_files:
        st.warning("Geen bestanden gevonden in upload folder, ga terug en upload opnieuw.")
        if st.button("Terug naar upload"):
            st.session_state.step = 1
            st.stop()
    else:

        if st.button("Leeg upload folder"):
            try:
                for filename in os.listdir(UPLOAD_DIR):
                    file_path = os.path.join(UPLOAD_DIR, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        st.warning(f"Kon {file_path} niet verwijderen: {e}")
                st.success("Upload folder is geleegd.")
            except Exception as e:
                st.warning(f"Kon upload folder niet legen: {e}")
        
        selected_file = st.selectbox("Selecteer bestand om te embedden", uploaded_files)

        client = QdrantClient(url="http://localhost:6333")
        existing_collections = [c.name for c in client.get_collections().collections]
        mode = st.radio(
            "Nieuwe collectie of toevoegen aan bestaande?",
            ["Nieuwe collectie", "Bestaande collectie"],
            index=0,
        )
        if mode == "Nieuwe collectie":
            collection_name = st.text_input("Naam nieuwe collectie", value="mijn_collectie")
            create_new = True
        else:
            collection_name = st.selectbox("Selecteer collectie", existing_collections)
            create_new = False

        visibility = st.radio("Classificatie", ["public", "private"], index=1)

        # Only run embedding if button pressed or embedding_running is True
        if st.button("Start embedding") or st.session_state.get("embedding_running", False):
            st.session_state.embedding_running = True
            if not collection_name:
                st.warning("Geef een collectie naam op.")
                st.session_state.embedding_running = False
                st.stop()
            try:
                checkpoint_file = os.path.join("checkpoints", f"{collection_name}_checkpoint.json")
                os.makedirs("checkpoints", exist_ok=True)

                if create_new and not client.collection_exists(collection_name):
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
                    )

                with open("config/config.json") as f:
                    config = json.load(f)
                embedding_model = OllamaEmbeddings(model=config.get("embedding_model", "nomic-embed-text"))
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

                processed_hashes = set()
                if os.path.exists(checkpoint_file):
                    with open(checkpoint_file, "r") as f:
                        processed_hashes = set(json.load(f))
                    st.info(f"{len(processed_hashes)} hashes geladen uit checkpoint")

                file_path = os.path.join(UPLOAD_DIR, selected_file)

                def process_file(filepath):
                    ext = os.path.splitext(filepath)[-1].lower()
                    docs = []
                    if ext == ".pdf":
                        docs = PyPDFLoader(filepath).load()
                    elif ext == ".csv":
                        docs = CSVLoader(file_path=filepath).load()
                    elif ext in [".txt", ".md"]:
                        docs = TextLoader(file_path=filepath).load()
                    elif ext == ".docx":
                        docs = UnstructuredWordDocumentLoader(file_path=filepath).load()
                    elif ext == ".json":
                        with open(filepath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            docs = (
                                [Document(page_content=json.dumps(r)) for r in data]
                                if isinstance(data, list)
                                else [Document(page_content=json.dumps(data))]
                            )
                    elif ext == ".xlsx":
                        import pandas as pd
                        df = pd.read_excel(filepath)
                        docs = [Document(page_content=row.to_json()) for _, row in df.iterrows()]
                    else:
                        return []
                    return splitter.split_documents(docs)

                def hash_text(text):
                    return hashlib.md5(text.encode("utf-8")).hexdigest()

                chunks = process_file(file_path)
                total_chunks = len(chunks)
                points_to_add = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                start_time = time.time()

                for i, doc in enumerate(chunks):
                    text = doc.page_content
                    content_hash = hash_text(text)
                    if content_hash in processed_hashes:
                        progress_bar.progress(int((i + 1) / total_chunks * 100))
                        continue
                    vector = embedding_model.embed_query(text)
                    point = PointStruct(
                        id=content_hash,
                        vector=vector,
                        payload={
                            "content": text,
                            "hash": content_hash,
                            "source": os.path.basename(file_path),
                            "visibility": visibility,
                        },
                    )
                    points_to_add.append((point, content_hash))

                    # ETA in hh:mm:ss
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (i + 1)
                    remaining = total_chunks - (i + 1)
                    eta = int(avg_time * remaining)
                    eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))

                    progress_bar.progress(int((i + 1) / total_chunks * 100))
                    status_text.text(f"Verwerkt {i + 1} van {total_chunks} fragmenten — ETA: {eta_str}")

                st.write(f"🔍 {total_chunks} fragmenten gescand, {len(points_to_add)} nieuw voor Qdrant")

                # Upsert in batches to avoid Qdrant payload size limit
                def batch_upsert(points, batch_size=100):
                    for i in range(0, len(points), batch_size):
                        client.upsert(collection_name=collection_name, points=points[i:i+batch_size])
                        time.sleep(0.05)  # kleine pauze om server te ontlasten

                if points_to_add:
                    batch_upsert([p[0] for p in points_to_add])
                    st.success(f"✅ {len(points_to_add)} punten toegevoegd aan collectie '{collection_name}'")

                    processed_hashes.update([h for _, h in points_to_add])
                    with open(checkpoint_file, "w") as f:
                        json.dump(list(processed_hashes), f)
                    st.info("💾 Checkpoint bijgewerkt")

                st.markdown("### 🔍 Voorbeeldfragmenten")
                for i, (point, _) in enumerate(points_to_add[:3]):
                    st.markdown(f"**Fragment {i + 1}**")
                    st.code(point.payload["content"][:500] + "..." if len(point.payload["content"]) > 500 else point.payload["content"])

                try:
                    shutil.rmtree(UPLOAD_DIR)
                    os.makedirs(UPLOAD_DIR, exist_ok=True)
                    st.info("Upload folder is geleegd.")
                except Exception as e:
                    st.warning(f"Kon upload folder niet legen: {e}")
            finally:
                st.session_state.embedding_running = False
else:
    st.info("Upload een bestand en ga verder.")