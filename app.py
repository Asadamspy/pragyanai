import streamlit as st
from rag_engine import load_data, build_vector_db, retrieve_colleges

st.set_page_config(page_title="PragyanAI", layout="wide")
st.title("🎓 PragyanAI – College Decision Intelligence")

# ---------- CACHE DATA & VECTOR DB ----------
@st.cache_data
def get_data():
    return load_data()

@st.cache_resource
def get_index(df):
    return build_vector_db(df)

# Load data and index (ONLY ONCE)
df = get_data()
index = get_index(df)

# ---------- STUDENT PROFILE ----------
st.sidebar.header("👤 Student Profile")

rank = st.sidebar.number_input("CET Rank", min_value=1, max_value=200000)
category = st.sidebar.selectbox("Category", ["General", "OBC", "SC", "ST"])
branch = st.sidebar.selectbox("Dream Branch", ["CSE", "ISE", "AI"])
city = st.sidebar.selectbox("City Preference", ["Bangalore", "Mysore"])
seat = st.sidebar.selectbox("Seat Preference", ["Govt", "Management"])

question = st.text_input(
    "Ask your question",
    "Which colleges are suitable for my rank?"
)

# ---------- RECOMMENDATION ----------
if st.button("🔍 Get Recommendation"):
    query = (
        f"Colleges for CET rank {rank} "
        f"category {category} "
        f"branch {branch} "
        f"city {city} "
        f"seat {seat}"
    )

    results = retrieve_colleges(query, df, index)

    st.subheader("📊 Recommendation Result")

    safe, moderate, stretch = [], [], []

    for _, row in results.iterrows():
        if rank <= row["cutoff_rank"]:
            safe.append(row)
        elif rank <= row["cutoff_rank"] + 2000:
            moderate.append(row)
        else:
            stretch.append(row)

    def show(title, data, emoji):
        st.markdown(f"### {emoji} {title}")
        if data:
            for r in data:
                st.success(
                    f"{r['college']} | {r['branch']} | "
                    f"{r['seat_type']} | Fees ₹{r['fees']}"
                )
        else:
            st.info("No colleges found")

    show("Safe Colleges", safe, "🟢")
    show("Moderate Colleges", moderate, "🟡")
    show("Stretch Colleges", stretch, "🔴")
