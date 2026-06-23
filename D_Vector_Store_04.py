import pickle       # Used for saving and loading Python objects to/from files
import numpy as np  # As we know that numpy is called numerical python and is used for high-performance numerical array operations
import faiss        # Meta's library for fast, efficient vector similarity search



# STEP 1: Load text chunks generated during preprocessing

def load_chunk():
    # Open pickle file containing chunked text data
    with open('Chunks.pkl', 'rb') as file:
        data = pickle.load(file)  # Deserialize file data back into a Python list

    # Return list of text chunks (e.g., ["text from page 1", "text from page 2"])
    return data




# STEP 2: Load embedding vectors from NumPy file

def load_embeddings():
    # Load the pre-computed numerical vector matrix from disk
    # Each row represents the embedding of one text chunk from Step 1
    return np.load('Embeddings.npy')




# STEP 3: Create and build FAISS index

def build_faiss(embeddings):

    # Unpack array shape to get total vectors (rows) and vector size (columns)
    # Example shape: (21, 384) -> 21 chunks, each with 384 dimensions
    num_chunks, dimension = embeddings.shape

    # FAISS works efficiently with float32 datatype
    embeddings_32 = embeddings.astype('float32')

    # Initialize a Flat Index using L2 (Euclidean) distance formula to measure similarity when the user asks a question.
    # Higher dimension values mean longer, more complex vectors
    index = faiss.IndexFlatL2(dimension)

    # Insert the 32-bit embedding vectors into the FAISS index structure
    index.add(embeddings_32)

    # Display total vectors successfully stored and ready for searching
    print(f"Total indexed chunks: {index.ntotal}")

    return index



# STEP 4: Save FAISS index and chunks to disk

def save_index(index, chunk):

    # FAISS index cannot be directly pickled on a native system
    with open('faiss_index.pkl', 'wb') as file:
        # Convert it into serialized binary array format first
        pickle.dump(faiss.serialize_index(index), file)
    print('File faiss_index.pkl saved successfully.')

    # Save the original text chunks to a separate file
    # Required later to retrieve actual text after search
    with open('chunks_texts.pkl', 'wb') as file:
        pickle.dump(chunk, file)
        print('File chunks_texts.pkl saved successfully.')



# STEP 5: Load saved FAISS index and chunks

def load_index():

    # Load serialized index from disk in read-binary mode
    with open('faiss_index.pkl', 'rb') as file:

        # Convert serialized binary back to FAISS index
        index = faiss.deserialize_index(pickle.load(file))
        print('File faiss_index.pkl loaded successfully.')

        # Open and load the companion file containing the raw text strings
    with open('chunks_texts.pkl', 'rb') as file:
        chunks = pickle.load(file)
        print('File chunks_texts.pkl loaded successfully.')

    # Return the searchable index and text lookup list together
    return index, chunks


# -------------------------------------------------------------------------------------------------------------------- 
# MAIN EXECUTION
# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Load chunked text data that is Fetch preprocessed text segments
    chunks = load_chunk()

    # 2. Fetch array matrix, convert to float32, and build the FAISS index
    index = build_faiss(load_embeddings())

    # 3. Permanently write both files to disk for future querying
    save_index(index, chunks)

    # 4. Verify integrity by loading the saved files back into memory
    load_index()




# ====================================================================================================
# ARCHITECTURE SUMMARY:
# Transforms a raw NumPy embedding matrix into a highly optimized, searchable FAISS vector index.
# In your current setup, this maps a (21, 384) matrix—representing 21 text chunks embedded into a 384-dimensional semantic space.

# LINE-BY-LINE MECHANICS & UNDER-THE-HOOD LOGIC:

# 1. Shape Extraction (embeddings.shape):
#    - Unpacks the matrix grid. Extracts 'num_chunks' (21 rows) and 'dimension' (384 columns). 
#    - The dimension configuration tells FAISS exactly how many coordinate points each vector has.

# 2. Type Casting (.astype('float32')):
#    - Crucial optimization step. Standard Python floats are 64-bit, but FAISS uses optimized C++
#    - SIMD (Single Instruction, Multiple Data) assembly instructions that require 32-bit floats.
#    - This downcasting slashes RAM usage in half and accelerates distance calculations exponentially.

# 3. Index Initialization (faiss.IndexFlatL2):
#    - Instantiates a "Flat" brute-force index using L2 (Euclidean) distance as the metric.
#    - "Flat" means it stores the raw vectors exactly as they are without lossy compression.
#    - "L2 Distance" calculates the straight-line geometric distance between vector coordinates.
#    - Closer distance = higher contextual and semantic similarity between your text chunks.

# 4. Vector Population (index.add):
#    - injects your 21 vectors into RAM-aligned contiguous memory arrays managed by the FAISS index.
#    - Once loaded, 'index.ntotal' will return 21, confirming the vector database is armed for querying.
# ====================================================================================================
