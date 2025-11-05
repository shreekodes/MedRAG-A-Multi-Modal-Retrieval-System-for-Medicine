# src/generation.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from retrieval import (
    load_faiss_index,
    load_metadata,
    search_text,
    TEXT_INDEX_PATH,
    TEXT_EMB_PATH,
    TEXT_META_PATH
)

# LOAD SUMMARIZATION MODEL

MODEL_NAME = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
-
# GENERATE SUMMARY

def generate_summary(papers, query):
    """
    Generate a concise summary based on top retrieved text papers.
    """
    context = ""
    for p in papers:
        context += f"Title: {p.get('title', '')}\n"

    prompt = (
        f"Based on the following research papers, answer the question:\n"
        f"{query}\n\n{context}\n\nSummarize clearly in 5 lines."
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_length=250, temperature=0.7, top_p=0.9)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


# MANUAL TESTING

if __name__ == "__main__":
    print("Loading FAISS index and metadata for test...")
    index = load_faiss_index(TEXT_INDEX_PATH, TEXT_EMB_PATH)
    meta = load_metadata(TEXT_META_PATH)

    while True:
        query = input("\n Enter a medical question (or 'exit'): ").strip()
        if query.lower() == "exit":
            break

        papers = search_text(query, index, meta, top_k=5)
        answer = generate_summary(papers, query)
        print("\n Answer:\n", answer)

