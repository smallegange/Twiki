import streamlit as st
from qdrant_client import QdrantClient
import pandas as pd

st.set_page_config(page_title="Qdrant Collectiebeheer", layout="wide")
st.title("Qdrant Collectiebeheer")

url = st.text_input("URL van Qdrant server", value="http://localhost:6333")

@st.cache_resource
def get_qdrant_client(u=None):
    if u:
        return QdrantClient(url=u)
    else:
        raise ValueError("Geen geldige connectiegegevens")

try:
    client = get_qdrant_client(u=url)
except Exception as e:
    st.error(f"❌ Fout bij verbinden met Qdrant: {e}")
    st.stop()

try:
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
except Exception as e:
    st.error(f"❌ Kan collecties niet ophalen: {e}")
    st.stop()

if not collection_names:
    st.success("✅ Geen collecties aanwezig.")
else:
    st.subheader("Bestaande collecties")
    selected = st.selectbox("Kies een collectie", collection_names, key="select_collection")

    if st.button(f"Verwijder '{selected}'"):
        try:
            client.delete_collection(collection_name=selected)
            st.success(f"✅ Collectie '{selected}' succesvol verwijderd.")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"❌ Fout bij verwijderen: {e}")

    # Data ophalen en in session_state bewaren
    if st.button("📄 Bekijk en filter inhoud"):
        try:
            scroll_result, _ = client.scroll(collection_name=selected, limit=5000)
            records = [point.payload | {"id": str(point.id)} for point in scroll_result]
            if records:
                df = pd.DataFrame(records)
                st.session_state["qdrant_df"] = df
                st.success(f"{len(df)} records geladen.")
            else:
                st.session_state["qdrant_df"] = pd.DataFrame()
                st.info("ℹ️ Geen data in deze collectie.")
        except Exception as e:
            st.error(f"❌ Kan inhoud niet laden of verwerken: {e}")

    # Alleen verder als er data is
    if "qdrant_df" in st.session_state and not st.session_state["qdrant_df"].empty:
        df = st.session_state["qdrant_df"]

        if "visibility" in df.columns:
            visibility_counts = df["visibility"].value_counts().to_dict()
            st.info(f"Zichtbaarheid: {visibility_counts}")
        else:
            st.warning("⚠️ Geen visibility-labels aanwezig. Voeg deze toe via de editor.")

        filter_col = st.selectbox("Filterkolom kiezen", df.columns, key="filter_col")
        unique_vals = df[filter_col].dropna().unique()
        selected_filter = st.selectbox("Waarde filteren", unique_vals, key="filter_val")
        filtered_df = df[df[filter_col] == selected_filter]

        st.dataframe(filtered_df)

        # Bulk aanpassing visibility
        st.markdown("### Bulk aanpassing visibility")
        if 'visibility' in df.columns:
            bulk_visibility = st.selectbox("Nieuwe visibility instellen voor alle gefilterde records", ["public", "private"], key="bulk_visibility")
            if st.button("Pas visibility toe op gefilterde selectie"):
                for row_id in filtered_df['id']:
                    client.set_payload(
                        collection_name=selected,
                        payload={"visibility": bulk_visibility},
                        points=[row_id]
                    )
                st.success(f"✅ Visibility aangepast naar '{bulk_visibility}' voor selectie.")
                st.experimental_rerun()

        # Bewerken
        st.markdown("**Bewerkbare tabel** (inclusief 'visibility')")
        edited_df = st.data_editor(filtered_df, num_rows="dynamic", key="edit_table")
        if st.button("Opslaan (overschrijft payloads)"):
            for _, row in edited_df.iterrows():
                payload = row.drop("id").to_dict()
                client.set_payload(
                    collection_name=selected,
                    payload=payload,
                    points=[row["id"]]
                )
            st.success("✅ Payloads bijgewerkt.")
            st.experimental_rerun()

        # Downloadknop
        if not filtered_df.empty:
            csv = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download gefilterde data als CSV",
                data=csv,
                file_name=f"{selected}_filtered.csv",
                mime="text/csv"
            )