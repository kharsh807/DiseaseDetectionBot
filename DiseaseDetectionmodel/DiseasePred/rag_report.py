from transformers import pipeline
from sentence_transformers import SentenceTransformer
import pdfplumber
import io

# Step 1: Extract text from PDF reports
def extract_text_from_pdf(file):
    with pdfplumber.open(io.BytesIO(file.read())) as pdf:
        text = "".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

# Step 2: Split text into smaller chunks
def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Step 3: Retrieve relevant chunks using embeddings
def retrieve_chunks(query, chunks, embedding_model, top_k=3):
    chunk_embeddings = embedding_model.encode(chunks)
    query_embedding = embedding_model.encode([query])
    similarities = (query_embedding @ chunk_embeddings.T).squeeze()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# Step 4: Summarize the retrieved chunks
def summarize_chunks(chunks, summarization_pipeline, max_length=512):
    summaries = []
    for chunk in chunks:
        if len(chunk.split()) > max_length:
            chunk = " ".join(chunk.split()[:max_length])
        try:
            summary = summarization_pipeline(chunk, max_length=100, min_length=30, do_sample=False)
            summaries.append(summary[0]["summary_text"])
        except Exception as e:
            summaries.append("Error processing this chunk.")
    return " ".join(summaries)

def iterative_summarization(text, summarization_pipeline, chunk_size=300, max_iterations=3):
    current_text = text
    for _ in range(max_iterations):
        chunks = chunk_text(current_text, chunk_size)
        current_text = summarize_chunks(chunks, summarization_pipeline)
        if len(current_text.split()) <= chunk_size:
            break
    return current_text

# Main function to process PDF and return summary
def process_pdf(file, query):
    embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    summarization_pipeline = pipeline("summarization", model="t5-small", tokenizer="t5-small")
    
    text = extract_text_from_pdf(file)
    chunks = chunk_text(text, chunk_size=300)
    relevant_chunks = retrieve_chunks(query, chunks, embedding_model)
    summary = summarize_chunks(relevant_chunks, summarization_pipeline)
    return summary
