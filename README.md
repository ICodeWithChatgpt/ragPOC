# 🧠 RAG-POC: Retrieval-Augmented Generation Proof of Concept

This project demonstrates a **Retrieval-Augmented Generation (RAG)** architecture, where vectorized content is stored, retrieved, and combined with OpenAI's GPT for generating accurate, context-aware responses.  

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [How It Works](#how-it-works)
---

## 🧐 Overview

In a RAG system, pre-indexed content is stored in a database and queried based on similarity to a user prompt. This project:
1. **Vectorizes textual data** using embeddings.
2. **Retrieves context** from a database based on cosine similarity.
3. Passes relevant context to GPT for **enhanced, context-aware answers**.

Use case: This approach avoids hallucinations and ensures generated responses are grounded in real data.

---

## 🚀 Features

- **Semantic Search**: Retrieves relevant content using cosine similarity on vector embeddings.
- **Database-Backed Storage**: Vectorized content is stored in SQLite.
- **OpenAI API Integration**: Combines retrieved data with GPT's generative capabilities.
- **Content Normalization**: Normalizes text for efficient vectorization and search.
- **Customizable Threshold**: Set similarity thresholds for better query results.

---

## 🛠️ Technologies Used

- **Python 3.9+**
- **Flask**: Lightweight backend server.
- **SQLite**: Simple database for vectorized content storage.
- **OpenAI API**: For generating embeddings and content generation.
- **NumPy**: For cosine similarity calculations.
- **Requests**: To interact with external APIs.

---

## ⚙️ Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone git@github.com:ICodeWithChatgpt/ragPOC.git


2. **Add your API Key**:

   - Get your OpenAI API key from [OpenAI](https://platform.openai.com/docs/guides/authentication).
   - Add your API key to the `.env` file.

## 🤖 How It Works
- In order to store content in the DB, run the main.py file.
  - It accepts both Raw text and URL as input.
  - The site or text is scraped
  - Metadata is extracted
  - Content is cleaned and vectorized

- In order to perform a search, run the app.py file.
  - It accepts a query as input.
  - The query is vectorized
  - If the button "Search in DB first" is checked, context is retrieved from the DB.
  - The cosine similarity is calculated between the query and the content in the DB
  - The content with the highest similarity is returned
  - The query, along with the retrieved content, is passed to the GPT model for generation.