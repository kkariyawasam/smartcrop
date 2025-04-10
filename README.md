# 🌿 Agro Yield Advisor

## Overview

The **Agro Yield Advisor** is a data-driven application that provides crop recommendations based on soil and climate data. This system leverages a combination of machine learning, natural language processing (NLP), and vector search to deliver accurate and tailored crop yield predictions.

The application integrates multiple AI models and an ensemble approach to suggest suitable crops for a given region based on its soil description, climate conditions (e.g., rainfall, soil pH), and historical agricultural data.

## UI Preview

## Key Features

- **AgroRAGAgent**: Uses a combination of vector embeddings and large language models (LLM) to search for similar regions and suggest suitable crops based on climate and soil similarities.
- **ExpertAgent**: Leverages an LLM to provide expert-level recommendations on suitable crops based on the region’s soil and climate.
- **RandomForestAgent**: Implements a machine learning model (Random Forest Classifier) to predict crop yield based on rainfall and soil pH values.
- **EnsembleAgent**: Combines results from **AgroRAGAgent**, **ExpertAgent**, and **RandomForestAgent** to provide a final recommendation using a majority voting mechanism.
- **Gradio Interface**: A user-friendly UI to input region descriptions, rainfall, and soil pH, and receive crop recommendations in markdown format.

## Technologies Used

- **Python**: The primary programming language used for development.
- **SentenceTransformers**: For generating sentence embeddings to compare region descriptions.
- **ChromaDB**: A vector database for storing and querying soil and climate data.
- **OpenAI GPT-4**: Used for NLP-based expert recommendations and agro-related suggestions.
- **scikit-learn**: Implements the Random Forest Classifier for crop yield prediction based on features like rainfall and pH.
- **Gradio**: For creating an interactive web interface for users to input data and get crop recommendations.
- **dotenv**: Loads environment variables, particularly for securely handling OpenAI API keys.
