🧠 Sentence Clusterer
💡 Motivation

Open-ended survey questions often produce a wide variety of responses that are difficult to encode or categorize automatically. Traditional text encoding methods fail to capture the semantic meaning of the responses, making analysis tedious and time-consuming.

This project aims to solve that problem by using semantic embeddings and cosine similarity to automatically group survey responses that express similar ideas — even if the wording is different.

✨ What It Does

Generates semantic embeddings for each response using Google’s Gemini Embedding API (gemini-embedding-001)

Calculates cosine similarity between responses to measure semantic closeness

Clusters similar responses automatically (based on a similarity threshold, e.g. ≥ 0.85)

Optionally uses Gemini-2.5-Flash to summarize each cluster into a short, meaningful sentence

⚙️ Technologies Used

Python

Google Generative AI (Gemini API)

NumPy

scikit-learn

🚀 Example Use Cases

Automatically grouping open-ended survey answers

Summarizing customer feedback into meaningful topics

Identifying repeated or similar responses in research questionnaires

🧩 How It Works

Convert each sentence into a vector using Gemini embeddings.

Compute pairwise cosine similarities.

Group sentences with high semantic similarity.

Summarize each group using Gemini to produce a clear interpretation.
