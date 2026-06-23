from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np



# TASK 1: Load FAISS index and stored chunk texts

def load_index():

    # Load serialized FAISS index from disk
    with open('faiss_index.pkl', 'rb') as file:
        index = faiss.deserialize_index(pickle.load(file))

    # Load original text chunks
    with open('chunks_texts.pkl', 'rb') as file:
        chunks_texts = pickle.load(file)

    # Return both index and text data
    return index, chunks_texts



# TASK 2: Load Sentence Transformer Model


# Pretrained model used to convert text into embeddings
# Model dimension = 384
model = SentenceTransformer('all-MiniLM-L6-v2')



# TASK 3: Perform Semantic Search

def semantic_search(question, index, chunks_texts):
    """
    Semantic Search Workflow

    1. Accept user question.
    2. Convert question into embedding vector.
    3. Search FAISS index for nearest vectors.
    4. Retrieve corresponding text chunks.
    5. Return most relevant chunks.
    """

    # Step 1: Convert user question into embedding
    # -------------------------------------------------

    # Model returns a vector representation of question
    question_vector = model.encode([question]).astype(np.float32)


    # Step 2: Search FAISS Index: k = 3 means retrieve top 3 most similar chunks
    # ---------------------------------------------------------------------------
    # distance -> similarity score (smaller = better match)
    # indices  -> positions of matching chunks
    distance, indices = index.search(question_vector, k=3)


    # Step 3: Retrieve actual chunk texts
    # -------------------------------------------------

    # Use returned indices to fetch text chunks
    results = [chunks_texts[i] for i in indices[0]]

    # Return top matching chunks
    return results



# TASK 4: Display Search Results

def display_results(results):

    print("\nTop Matching Chunks:")

    # Print chunks with numbering
    for i, chunk in enumerate(results, 1):
        print(f"{i}. {chunk}\n")



#----------------------------------------------------------#----------------------------------------------------------#----------------------------------------------------------#

# MAIN EXECUTION
if __name__ == "__main__":

    # Load FAISS index and chunk texts
    index, chunks_texts = load_index()

    # Take question from user
    user_question = input("Enter your question: ")

    # Perform semantic search
    search_results = semantic_search(
        user_question,
        index,
        chunks_texts
    )

    # Display retrieved chunks
    display_results(search_results)