# src/generation.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from retrieval import build_faiss_index, load_metadata, search_papers

# Load the model
MODEL_NAME = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def generate_summary(papers, query):
    # Combine top 5 paper abstracts into one context
    context = ""
    for p in papers:
        context += f"Title: {p.get('title','')}\n"

    # Create the input prompt
    prompt = f"Based on the following research papers, answer the question:\n{query}\n\n{context}\n\nSummarize clearly in 5 lines."

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_length=250, temperature=0.7, top_p=0.9)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


if __name__ == "__main__":
    index = build_faiss_index()
    meta = load_metadata()

    while True:
        query = input("\n💬 Enter a medical question (or 'exit'): ").strip()
        if query.lower() == "exit":
            break

        papers = search_papers(query, index, meta)
        answer = generate_summary(papers, query)
        print("\n🧠 Answer:\n", answer)
