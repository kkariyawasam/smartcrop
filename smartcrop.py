# #!/usr/bin/env python
# # coding: utf-8

# # In[1]:


# # # regional soil + climate descriptions into embeddings(list of numbers that a computer can understand and compare) and store 
# # #them in ChromaDB for retrieval

# # STEP 1: Import Libraries
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import chromadb
# from chromadb.config import Settings

# # STEP 2: Load Data (limit to avoid crashes)
# df = pd.read_csv("data/soil_climate_data.csv").head(10)  # Change head(10) to full later

# # STEP 3: Combine text fields into one for embedding
# df["combined"] = df["soil_description"] + ". " + df["climate_notes"]

# # STEP 4: Load lightweight embedding model
# model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# # STEP 5: Generate embeddings
# texts = df["combined"].tolist()
# embeddings = model.encode(texts)

# # STEP 6: Setup ChromaDB client
# client = chromadb.PersistentClient(path="data/chroma_db")  # persists to disk

# # STEP 7: Create or get collection
# collection = client.get_or_create_collection(name="soil_embeddings")

# # STEP 8: Add documents to the collection
# for i, (text, emb) in enumerate(zip(texts, embeddings)):
#     collection.add(
#         ids=[str(i)],
#         documents=[text],
#         embeddings=[emb.tolist()],
#         metadatas=[{"region": df["region"][i]}]
#     )

# print("✅ Step 1 complete: Embeddings stored in ChromaDB!")



# # In[2]:


# import subprocess
# subprocess.run(["pip", "install", "umap-learn"], shell=True)


# # In[3]:


# # Check if similar regions (like climates or soil types) cluster together in 2D space

# import chromadb
# import pandas as pd
# import umap
# import matplotlib.pyplot as plt

# # Connect to ChromaDB and fetch collection
# chroma_client = chromadb.PersistentClient(path="data/chroma_db")
# collection = chroma_client.get_collection(name="soil_embeddings")

# # Extract metadata and embeddings
# results = collection.get(include=["embeddings", "metadatas"])
# embeddings = results["embeddings"]
# regions = [meta["region"] for meta in results["metadatas"]]


# print("Number of embeddings:", len(embeddings))
# print("Example embedding shape:", embeddings[0].shape if len(embeddings) > 0 else "No embeddings")

# # Reduce dimensions
# reducer = umap.UMAP(n_neighbors=2, random_state=42)
# embedding_2d = reducer.fit_transform(embeddings)

# # Plot
# plt.figure(figsize=(10, 6))
# plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c='skyblue')
# for i, label in enumerate(regions):
#     plt.text(embedding_2d[i, 0], embedding_2d[i, 1], label)
# plt.title("Soil & Climate Embedding Visualization")
# plt.xlabel("UMAP-1")
# plt.ylabel("UMAP-2")
# plt.show()


# # In[4]:


# # This agent uses similar region data from ChromaDB + LLM for reasoning

# from sentence_transformers import SentenceTransformer
# import chromadb
# import openai
# from dotenv import load_dotenv
# import os
# from chromadb import Client

# chroma_client = Client()

# chroma_client = chromadb.PersistentClient(path="data/chroma_db")
# collection = chroma_client.get_collection(name="soil_embeddings")


# def agro_rag_agent(region_text):
#     # Step 1: Embed the input
#     embedder = SentenceTransformer("all-MiniLM-L6-v2")
#     embedding = embedder.encode(region_text).tolist()

#     # Step 2: Retrieve similar regions
#     results = collection.query(
#         query_embeddings=[embedding],
#         n_results=5  # top 5 similar regions
#     )

#     # Step 3: Build context
#     context = ""
#     for doc in results["documents"][0]:
#         context += f"- {doc}\n"

#     # Step 4: Ask LLM
#     prompt = f"""
# Context:
# {context}

# Region Description:
# {region_text}

# Question:
# Based on similar regions, what crops are suitable?
# """
#     # LLM call (OpenAI example)
#     load_dotenv()
#     api_key = os.getenv('OPENAI_API_KEY')
#     response = openai.chat.completions.create(
#     model="gpt-4o-mini",  # Use the correct model identifier
#     messages=[{"role": "user", "content": prompt}],
#     temperature=0.7
# )

# # Step 4: Print the result
#     return response.choices[0].message.content


# # In[5]:


# region_input = "Region with tropical climate, loamy soil, and 1800mm annual rainfall"

# print("🔍 AgroRAGAgent Recommendation:")
# print(agro_rag_agent(region_input))


# # In[6]:


# #This one doesn’t use vector retrieval — it directly predicts crops (Reasoning from documents)

# def expert_agent(region_text):
#     prompt = f"""
# You're an agronomy expert.

# Region Description:
# {region_text}

# Question:
# What crops are suitable for this region based on climate and soil?
# """
#     load_dotenv()
#     api_key = os.getenv('OPENAI_API_KEY')
#     response = openai.chat.completions.create(
#     model="gpt-4o-mini",  # Use the correct model identifier
#     messages=[{"role": "user", "content": prompt}],
#     temperature=0.7
# )

#     return response.choices[0].message.content


# # In[7]:


# print("\n🧠 ExpertAgent Recommendation:")
# print(expert_agent(region_input))


# # In[8]:


# #Build RandomForestAgent

# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split

# # Load data
# df = pd.read_csv("data/soil_climate_data.csv")

# # Features and target
# X = df[["rainfall", "pH"]]
# y = df["yield"]

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# rf_model = RandomForestClassifier()
# rf_model.fit(X_train, y_train)

# # Prediction wrapper
# def random_forest_agent(rainfall, ph):
#     return rf_model.predict([[rainfall, ph]])[0]


# # In[9]:


# #EnsembleAgent

# from collections import Counter

# def ensemble_agent(region_text, rainfall, ph):
#     # Get individual predictions
#     rag_result = agro_rag_agent(region_text)
#     expert_result = expert_agent(region_text)
#     rf_result = random_forest_agent(rainfall, ph)

#     print("AgroRAGAgent →", rag_result)
#     print("ExpertAgent →", expert_result)
#     print("RandomForestAgent →", rf_result)

#     # Simplify outputs (optional NLP parsing if LLMs return text)
#     def simplify(result):
#         if "rice" in result.lower(): return "High"
#         elif "millet" in result.lower(): return "Low"
#         else: return "Medium"

#     votes = [
#         simplify(rag_result),
#         simplify(expert_result),
#         rf_result
#     ]

#     final_vote = Counter(votes).most_common(1)[0][0]
#     return f"🌾 Final Recommendation: {final_vote} yield crops"


# # In[12]:


# region_text = "Tropical, loamy soil, 1800mm rainfall"
# rainfall = 1800
# ph = 6.5

# print("11"+ensemble_agent(region_text, rainfall, ph))


# # In[ ]:



#!/usr/bin/env python
# coding: utf-8

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

# === Step 2: Load Data and Setup Embeddings (Optional: You can skip rerunning this repeatedly) ===
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



