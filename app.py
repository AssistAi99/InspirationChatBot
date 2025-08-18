# app.py
import streamlit as st
import openai
import numpy as np
import json
import faiss
import pandas as pd

# ========== CONFIG ==========
openai.api_key = st.secrets["OPENAI_API_KEY"]
INDEX_FILE = "games_index.faiss"
DATA_FILE = "games_data.jsonl"

# ========== LOAD DATA ==========
index = faiss.read_index(INDEX_FILE)
with open(DATA_FILE, "r") as f:
    game_records = [json.loads(line) for line in f]

# Full Excel (for extra metadata)
df = pd.read_excel("NewInspired.xlsx")

# ========== HELPERS ==========
def get_embedding(text: str):
    response = openai.Embedding.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

def search_candidates(query: str, top_k=3, threshold=0.70):
    """Find top K candidates using FAISS. Only return if similarity above threshold."""
    query_emb = get_embedding(query)
    D, I = index.search(np.array([query_emb]).astype("float32"), top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        similarity = 1 / (1 + score)  # crude conversion (lower L2 → higher similarity)
        if similarity >= threshold:
            results.append((game_records[idx], similarity))
    return results

def ask_gpt_strict(query: str, top_candidate: dict, other_candidates: list):
    """GPT explains strictly based on candidates. No invention allowed."""
    context = f"Top candidate:\n- {top_candidate['Game Name']} (Publisher: {top_candidate['Publisher']}, Inspired by: {top_candidate['Inspiration']})\n"

    if other_candidates:
        context += "\nOther possible matches:\n"
        for c in other_candidates:
            context += f"- {c['Game Name']} (Publisher: {c['Publisher']})\n"

    prompt = f"""
The user asked: "{query}"

{context}

Rules:
-You must ONLY use the given candidates.
-Select the best-matching candidate to the user’s query (based on relevance). Users Query - {query}.
-Clearly state which game is the main recommendation and explain why it matches well.
-Mention that other candidates are also related, but emphasize the best match first.
-Never invent or add games that are not in the list.
-Respond in a friendly, conversational way.
"""

    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a strict game database assistant."},
                  {"role": "user", "content": prompt}]
    )
    return resp['choices'][0]['message']['content']

def enrich_with_excel(game_name):
    """Fetch all metadata for a given game from Excel."""
    row = df[df["Game Name"] == game_name]
    if row.empty:
        return None
    return row.iloc[0].to_dict()

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="Game Finder Chatbot", page_icon="🎮", layout="centered")

st.title("🎮 Game Finder AI")
st.write("Ask me about any game, and I'll find the closest match from my database!")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Type a game name or inspiration...")

if user_input:
    st.session_state.history.append(("user", user_input))

    candidates = search_candidates(user_input, top_k=3, threshold=0.70)

    if not candidates:  # No match found
        st.session_state.history.append(("bot", "❌ No game found in my database for that query."))
    else:
        top_candidate, _ = candidates[0]
        others = [c for c, _ in candidates[1:]]

        gpt_reply = ask_gpt_strict(user_input, top_candidate, others)
        st.session_state.history.append(("bot", gpt_reply))

        # Enrich with Excel full data (URL, Icon, Game Screenshots, etc.)
        game_data = enrich_with_excel(top_candidate["Game Name"])
        if game_data:
            # Show structured info (customizable)
            st.session_state.history.append(("meta", game_data))

# Render chat
for role, msg in st.session_state.history:
    if role == "user":
        st.chat_message("user").write(msg)
    elif role == "bot":
        st.chat_message("assistant").markdown(msg)
    elif role == "meta":
        st.subheader(msg["Game Name"])
        st.write(f"**Publisher:** {msg.get('Publisher', 'N/A')}")
        st.write(f"**Inspiration:** {msg.get('Inspiration', 'N/A')}")
        if "Game URL" in msg and pd.notna(msg["Game URL"]):
            st.markdown(f"🔗 [Play / Info Link]({msg['Game URL']})")
        if "Game Icon URL" in msg and pd.notna(msg["Game Icon URL"]):
            st.image(msg["Game Icon URL"], width=100)
        #if "Game Screenshots" in msg and pd.notna(msg["Game Screenshots"]):
        #    st.image(msg["Game Screenshots"], width=300)
         # ✅ Handle multiple screenshot links
        #if "Game Screenshots" in msg and pd.notna(msg["Game Screenshots"]):
        #    urls = str(msg["Game Screenshots"]).replace("\n", ",").replace(" ", ",").split(",")
        #    urls = [u.strip() for u in urls if u.strip()]
        #    if urls:
        #        st.write("📸 **Screenshots:**")
        #        cols = st.columns(min(3, len(urls)))  # show in rows of 3
        #        for i, url in enumerate(urls):
        #            with cols[i % 3]:
        #                st.image(url, width=200)  
         # ✅ Show up to 4 screenshots in a single row
        if "Game Screenshots" in msg and pd.notna(msg["Game Screenshots"]):
            urls = str(msg["Game Screenshots"]).replace("\n", ",").replace(" ", ",").split(",")
            urls = [u.strip() for u in urls if u.strip()]
            if urls:
                st.write("📸 **Screenshots:**")
                cols = st.columns(4)  # fixed 4 columns
                for i in range(min(4, len(urls))):  # only first 4 screenshots
                    with cols[i]:
                        st.image(urls[i], width=200)
        # Any extra columns will show automatically
        for col, val in msg.items():
            if col not in ["Game Name", "Publisher", "Inspiration", "Game URL", "Game Icon URL", "Game Screenshots"]:
                st.write(f"**{col}:** {val}")
