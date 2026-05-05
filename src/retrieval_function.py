def get_relevant_context(question, text_chunks, index, model):
    question_vec = model.encode([question])
    D, I = index.search(question_vec, k=3)
    
    relevant_chunks = [text_chunks[i] for i in I[0]]
    return " ".join(relevant_chunks)
