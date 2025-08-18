from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import openai
import numpy as np
import json
import faiss
import re
from difflib import SequenceMatcher


#openai.api_key = "YOUR_OPENAI_API_KEY"
openai.api_key = "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"


app = FastAPI()

# Allow frontend access (adjust if deploying)
app.add_middleware(
    CORSMiddleware,
    #allow_origins=["*"],  # replace with your frontend domain in production
    allow_origins=["http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load index and data
index = faiss.read_index("games_index.faiss")

def normalize(text):
    return re.sub(r"[^a-zA-Z0-9 ]", "", text.lower()).strip()

def is_similar(a, b, threshold=0.97):  # default stricter
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio() >= threshold

with open("games_data.jsonl", "r") as f:
    game_records = [json.loads(line) for line in f]

def get_embedding(text: str):
    response = openai.Embedding.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

@app.post("/searchOLD")
async def search_game(request: Request):
    body = await request.json()
    query = body.get("query", "")

    if not query.strip():
        return {"found": False, "suggestion": "Please enter a valid game name."}

    # Embed query
    query_embedding = get_embedding(query)
    D, I = index.search(np.array([query_embedding]).astype("float32"), 1)

    score = 1 - D[0][0] / 2  # cosine similarity estimate
    match_index = I[0][0]

    if score >= 0.85:
        match = game_records[match_index]
        return {
            "found": True,
            "game_name": match["Game Name"],
            "publisher": match["Publisher"],
            "inspiration": match["Inspiration"]
        }
    else:
        # Try suggesting inspiration-based fallback
        keywords = ["seat away", "bus jam", "subway surfer"]
        for k in keywords:
            if k.lower() in query.lower():
                return {
                    "found": False,
                    "suggestion": f"I don’t have a game like that. Want to explore games inspired by '{k.title()}'?"
                }
        return {
            "found": False,
            "suggestion": "❌ I don’t have a game like that. Try another?"
        }

@app.post("/searchOLD")
async def search_game(request: Request):
    body = await request.json()
    query = body.get("query", "").strip().lower()

    if not query:
        return {"found": False, "suggestion": "Please enter a valid game name."}

    # 1. Exact match
    # 1. Enhanced Exact Match
    exact_matches = [
        g for g in game_records if is_similar(g["Game Name"], query)
    ]
    if exact_matches:
        return {
            "found": True,
            "matches": [
                {
                    "game_name": g["Game Name"],
                    "publisher": g["Publisher"],
                    "inspiration": g["Inspiration"]
                }
                for g in exact_matches
            ]
        }
    # 2. Embedding-based semantic close matches
    query_embedding = get_embedding(query)
    D, I = index.search(np.array([query_embedding]).astype("float32"), 5)

    suggestions = []
    for distance, idx in zip(D[0], I[0]):
        score = 1 - distance / 2  # approx cosine sim
        if score >= 0.75:  # adjustable threshold
            game = game_records[idx]
            suggestions.append({
                "game_name": game["Game Name"],
                "publisher": game["Publisher"],
                "inspiration": game["Inspiration"]
            })

    if suggestions:
        return {
            "found": False,
            "suggestion": "I couldn't find an exact match. But here are some close matches I recommend:",
            "suggestions": suggestions
        }

    # 3. Nothing found at all
    return {
        "found": False,
        "suggestion": "❌ I couldn’t find a game with that name or anything close to it. Try a different query?"
    }

@app.post("/searchOLD")
async def search_game(request: Request):
    body = await request.json()
    query = body.get("query", "").strip().lower()

    if not query:
        return {"found": False, "suggestion": "Please enter a valid game name."}

    # 1. Enhanced Exact Match
    exact_matches = [
        g for g in game_records if is_similar(g["Game Name"], query)
    ]
    if exact_matches:
        return {
            "found": True,
            "matches": [
                {
                    "game_name": g["Game Name"],
                    "publisher": g["Publisher"],
                    "inspiration": g["Inspiration"]
                }
                for g in exact_matches
            ],
            "label": "✅ This is the most accurate match I found:"
        }

    # 2. Embedding-based semantic close matches
    query_embedding = get_embedding(query)
    D, I = index.search(np.array([query_embedding]).astype("float32"), 5)

    suggestions = []
    top_match = None

    for i, (distance, idx) in enumerate(zip(D[0], I[0])):
        score = 1 - distance / 2
        if score >= 0.75:
            game = game_records[idx]
            game_data = {
                "game_name": game["Game Name"],
                "publisher": game["Publisher"],
                "inspiration": game["Inspiration"]
            }
            if i == 0:
                top_match = game_data
            else:
                suggestions.append(game_data)

    if top_match:
        return {
            "found": False,
            "top_match": top_match,
            "top_label": "🤖 This is the closest matching game I found:",
            "suggestions": suggestions,
            "suggestion_label": "🧠 Other similar games:"
        }

    # 3. No match at all
    return {
        "found": False,
        "suggestion": "❌ I couldn’t find a game with that name or anything close to it."
    }

@app.post("/searchOLD")
async def search_game(request: Request):
    body = await request.json()
    query = body.get("query", "").strip().lower()

    if not query:
        return {"found": False, "suggestion": "Please enter a valid game name."}

    # Step 1: Enhanced Exact Match
    exact_matches = [g for g in game_records if is_similar(g["Game Name"], query, threshold=0.97)]
    if exact_matches:
        return {
            "found": True,
            "matches": [{
                "game_name": g["Game Name"],
                "publisher": g["Publisher"],
                "inspiration": g["Inspiration"]
            } for g in exact_matches],
            "label": "✅ This is the most accurate match I found:"
        }

    # Step 2: Semantic Embedding-Based Suggestions
    query_embedding = get_embedding(query)
    D, I = index.search(np.array([query_embedding]).astype("float32"), 5)

    suggestions = []
    top_match = None

    for i, (distance, idx) in enumerate(zip(D[0], I[0])):
        score = 1 - distance / 2
        if score >= 0.75:
            game = game_records[idx]
            entry = {
                "game_name": game["Game Name"],
                "publisher": game["Publisher"],
                "inspiration": game["Inspiration"]
            }
            if i == 0:
                top_match = entry
            else:
                suggestions.append(entry)

    if top_match:
        return {
            "found": False,
            "top_match": top_match,
            "top_label": "🤖 This is the closest matching game I found:",
            "suggestions": suggestions,
            "suggestion_label": "🧠 Other similar games you might like:"
        }

    return {
        "found": False,
        "suggestion": "❌ I couldn’t find any game related to your query."
    }
    
@app.post("/search")
async def search_game(request: Request):
    body = await request.json()
    query = body.get("query", "").strip().lower()

    if not query:
        return {"found": False, "suggestion": "Please enter a valid game name or inspiration."}

    def normalize(text):
        return re.sub(r"[^a-zA-Z0-9 ]", "", text.lower()).strip()

    def is_similar(a, b, threshold=0.97):
        return SequenceMatcher(None, normalize(a), normalize(b)).ratio() >= threshold

    exact_match = None
    for g in game_records:
        # Check both Game Name and Inspiration
        if is_similar(g["Game Name"], query) or is_similar(g["Inspiration"], query, threshold=0.92):
            exact_match = {
                "game_name": g["Game Name"],
                "publisher": g["Publisher"],
                "inspiration": g["Inspiration"]
            }
            break

    # Embedding-based fallback
    query_embedding = get_embedding(query)
    D, I = index.search(np.array([query_embedding]).astype("float32"), 6)

    suggestions = []
    top_match = None

    for i, (distance, idx) in enumerate(zip(D[0], I[0])):
        score = 1 - distance / 2
        if score >= 0.75:
            game = game_records[idx]
            data = {
                "game_name": game["Game Name"],
                "publisher": game["Publisher"],
                "inspiration": game["Inspiration"]
            }
            if i == 0:
                top_match = data
            else:
                suggestions.append(data)

    # Return exact match if found
    if exact_match:
        return {
            "found": True,
            "matches": [exact_match],
            "label": "✅ This is the most accurate match I found (based on name or inspiration):"
        }

    if top_match:
        return {
            "found": False,
            "top_match": top_match,
            "top_label": "🤖 This is the closest matching game I found:",
            "suggestions": suggestions,
            "suggestion_label": "🧠 Other similar games you might like:"
        }

    return {
        "found": False,
        "suggestion": "❌ I couldn’t find any game related to your query."
    }
