# === Step 1: Load Libraries ===
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
import openai
import os
from chromadb import Client
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from collections import Counter
import gradio as gr

# === Step 2: Load Data and Setup Embeddings ===
df = pd.read_csv("data/soil_climate_data.csv").head(10)
df["combined"] = df["soil_description"] + ". " + df["climate_notes"]
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
texts = df["combined"].tolist()
embeddings = model.encode(texts)

client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_or_create_collection(name="soil_embeddings")

# Optional: Populate collection only if not already filled (avoid duplicates)
if len(collection.get(include=["documents"])["documents"]) == 0:
    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        collection.add(
            ids=[str(i)],
            documents=[text],
            embeddings=[emb.tolist()],
            metadatas=[{"region": df["region"][i]}]
        )

# === Step 3: Define Agents ===

# AgroRAGAgent (LLM + Vector Similarity)
def agro_rag_agent(region_text):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = embedder.encode(region_text).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=5)

    context = ""
    for doc in results["documents"][0]:
        context += f"- {doc}\n"

    prompt = f"""
Context:
{context}

Region Description:
{region_text}

Question:
Based on similar regions, what crops are suitable?
"""
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

# ExpertAgent (LLM Direct)
def expert_agent(region_text):
    prompt = f"""
You're an agronomy expert.

Region Description:
{region_text}

Question:
What crops are suitable for this region based on climate and soil?
"""
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

# RandomForestAgent
df_full = pd.read_csv("data/soil_climate_data.csv")
X = df_full[["rainfall", "pH"]]
y = df_full["yield"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

def random_forest_agent(rainfall, ph):
    return rf_model.predict([[rainfall, ph]])[0]

# EnsembleAgent
def ensemble_agent(region_text, rainfall, ph):
    rag_result = agro_rag_agent(region_text)
    expert_result = expert_agent(region_text)
    rf_result = random_forest_agent(rainfall, ph)

    def simplify(result):
        if "rice" in result.lower(): return "High"
        elif "millet" in result.lower(): return "Low"
        else: return "Medium"

    votes = [
        simplify(rag_result),
        simplify(expert_result),
        rf_result
    ]

    final_vote = Counter(votes).most_common(1)[0][0]

    result_text = f"""
🌾 Final Recommendation: **{final_vote} yield crops**

🧠 ExpertAgent → {expert_result.strip()}

🔍 AgroRAGAgent → {rag_result.strip()}

🌲 RandomForestAgent → {rf_result}
"""
    return result_text

# === Step 4: Gradio UI ===

def run_ensemble(region_text, rainfall, ph):
    return ensemble_agent(region_text, rainfall, ph)

gr.Interface(
    fn=run_ensemble,
    inputs=[
        gr.Textbox(label="Region Description", placeholder="e.g., Tropical, loamy soil, 1800mm rainfall"),
        gr.Slider(label="Rainfall (mm)", minimum=0, maximum=3000, step=10, value=1800),
        gr.Slider(label="Soil pH", minimum=0.0, maximum=14.0, step=0.1, value=6.5),
    ],
    outputs="markdown",
    title="🌿 Agro Yield Advisor",
    description="Get crop recommendations based on soil & climate data using AI + ML + Vector Search."
).launch()



