# =========================
# 📌 AI Game Recommender (Streamlit + OpenAI + FAISS)
# =========================

import streamlit as st
import openai
import faiss
import numpy as np
import pandas as pd
import json

# ========== SETUP ==========
openai.api_key = st.secrets["OPENAI_API_KEY"]
INDEX_FILE = "games_index.faiss"
DATA_FILE = "games_data.jsonl"

# ========== LOAD DATA ==========
index = faiss.read_index(INDEX_FILE)
with open(DATA_FILE, "r") as f:
    game_records = [json.loads(line) for line in f]

# Full Excel (for extra metadata)
df = pd.read_excel("NewInspired.xlsx")


# ========== HELPER: Embedding ==========
# ========== HELPERS ==========
def get_embedding(text: str):
    response = openai.Embedding.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']


# ========== NEW: GPT to extract only closest synonyms ==========
def extract_keywords(query: str):
    prompt = f"""
    Extract only the core keywords and their closest synonyms from the user request.

    User query: "{query}"

    Rules:
    - Exclude the generic word "game" (ignore it completely).
    - Always include the exact words from the user query (don’t drop them).
    - Keep keywords short (1–2 words max).
    - Only include direct synonyms or very close related terms.
    - Do NOT expand too much or include irrelevant associations.
    - Return strict JSON in this format:
    {{
        "keywords": ["word1", "word2", "word3"]
    }}
    """

    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You extract concise keywords and very close synonyms only."},
            {"role": "user", "content": prompt}
        ]
    )
    reply = resp['choices'][0]['message']['content']
    print("GPT keyword output:", reply)

    try:
        return json.loads(reply).get("keywords", [])
    except:
        return []


# ========== IMPROVED PROGRESSIVE SEARCH ==========
from itertools import combinations

def search_with_keywords(keywords: list, top_k=5, threshold=0.70):
    seen = {}

    # 1️⃣ Single keyword search
    for kw in keywords:
        query_emb = get_embedding(kw)
        D, I = index.search(np.array([query_emb]).astype("float32"), top_k)
        for score, idx in zip(D[0], I[0]):
            if idx == -1: continue
            similarity = 1 / (1 + score)
            if similarity >= threshold:
                if idx not in seen or similarity > seen[idx]["similarity"]:
                    seen[idx] = {"record": game_records[idx], "similarity": similarity}

    # 2️⃣ Pair combinations
    for combo in combinations(keywords, 2):
        phrase = " ".join(combo)
        query_emb = get_embedding(phrase)
        D, I = index.search(np.array([query_emb]).astype("float32"), top_k)
        for score, idx in zip(D[0], I[0]):
            if idx == -1: continue
            similarity = 1 / (1 + score)
            if similarity >= threshold - 0.05:  # looser for pairs
                if idx not in seen or similarity > seen[idx]["similarity"]:
                    seen[idx] = {"record": game_records[idx], "similarity": similarity}

    # 3️⃣ Triplet combinations
    if len(keywords) >= 3:
        for combo in combinations(keywords, 3):
            phrase = " ".join(combo)
            query_emb = get_embedding(phrase)
            D, I = index.search(np.array([query_emb]).astype("float32"), top_k)
            for score, idx in zip(D[0], I[0]):
                if idx == -1: continue
                similarity = 1 / (1 + score)
                if similarity >= threshold - 0.1:  # even looser for triplets
                    if idx not in seen or similarity > seen[idx]["similarity"]:
                        seen[idx] = {"record": game_records[idx], "similarity": similarity}

    # 4️⃣ Full query search (as backup)
    full_query = " ".join(keywords)
    query_emb = get_embedding(full_query)
    D, I = index.search(np.array([query_emb]).astype("float32"), top_k)
    for score, idx in zip(D[0], I[0]):
        if idx == -1: continue
        similarity = 1 / (1 + score)
        if similarity >= threshold:
            if idx not in seen or similarity > seen[idx]["similarity"]:
                seen[idx] = {"record": game_records[idx], "similarity": similarity}

    # ✅ Final results sorted
    results = sorted(seen.values(), key=lambda x: x["similarity"], reverse=True)
    print("Final search results:", results)
    return [r["record"] for r in results]





# ========== NEW: GPT to pick best match ==========
def generate_final_answer(query: str, candidates: list):
    if not candidates:
        # polite no-result answer
        return "🙇 Sorry, I couldn’t find any game that matches your query. Maybe try different words?"

    context = "Here are the candidate games from database:\n"
    for c in candidates:
        context += f"- {c['Game Name']} (Publisher: {c['Publisher']}, Inspired by: {c['Inspiration']}, Link: {c.get('Game URL','N/A')})\n"

    print(candidates)
    prompt = f"""
    User query: "{query}"

    {context}

    Rules:
    - Pick ONE game as the top recommendation if it clearly matches user's intent.
    - Show it first with explanation.
    - Then list other related games.
    - If no strong match, say politely: "I’m not sure about an exact match, but here are some similar games..."
    - Output in friendly conversational style.
    - Do NOT invent games outside given list.
    - Show only Game Name, Publisher, Inspiration, and AppStore/PlayStore link if available.
    """

    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful, polite game recommender assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    print(resp['choices'][0]['message']['content'])
    return resp['choices'][0]['message']['content']


# =========================
# 🚀 STREAMLIT APP
# =========================
st.set_page_config(page_title="AI Game Recommender", page_icon="🎮", layout="centered")

st.title("🎮 AI Game Recommender")
st.write("Ask me about any type of mobile game and I’ll suggest the best match!")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Type your game request...")

if user_input:
    st.session_state.history.append(("user", user_input))

    # Step 1: Extract keywords
    keywords = extract_keywords(user_input)

    # Step 2: Search embeddings
    candidates = search_with_keywords(keywords, top_k=5)

    # Step 3: Generate final GPT answer
    bot_message = generate_final_answer(user_input, candidates)

    # Step 4: Save bot response
    st.session_state.history.append(("bot", bot_message))


# Render chat
for role, msg in st.session_state.history:
    if role == "user":
        st.chat_message("user").write(msg)
    elif role == "bot":
        st.chat_message("assistant").markdown(msg)
