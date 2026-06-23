"""
Day-02

Goal:
1. Read extracted text from a file
2. Split the large text into smaller pieces (chunks)
3. Save chunks for later usage

Why chunking?
LLMs cannot process extremely large text at once, so we divide text into
smaller manageable pieces called chunks.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter   # for chunking text
import pickle    # to save Python objects (.pkl files)


# Task 1: Load the extracted.txt file in read mode.
def load_file():
    with open('extract_text.txt') as file:   # Open text file in read mode studied during file handling in python.
        text = file.read()                  # Read complete file content
        return text
    


# Task 2: Chunking the raw text from extracted.txt file in the form of chunks .
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,            # Maximum characters allowed in one chunk
        chunk_overlap=50)          # Overlap between consecutive chunks helps preserve context between chunks such that relationship of meaning between chunks remain preserved.

    chunks = splitter.split_text(text)    # Perform chunking
    return chunks                         # return kardenge list of chunks



# Print top 3 chunks, Utility function to display chunks
def view_chunks(chunks):
    for index , value in enumerate(chunks):       # enumerate() gives index + value together
        print(f'Chunk {index} : {value}')
        print('-----------------------------')

# Saving all the chunks in a .txt file named as chunks.txt - Useful for manually viewing chunks
def save_chunks(chunks):
    with open('Chunks.txt','w') as file:     # Open file in write mode
        for index, value in enumerate(chunks):   # Enumerate: Allows you to track both the index position and the item value simultaneously during a loop.
            file.write(f'Chunk {index} : {value}\n')
            file.write('-----------------------------\n')



# Task 3: Save the chunks in .pkl format. 

# Pickle stores Python objects directly and Faster to load later compared to re-processing

def save_chunks_pickle(chunks):
    with open('Chunks.pkl','wb') as file:          #wb -> write in bytes or  wb -> write binary mode
        pickle.dump(chunks, file)                          # dump-> Save the chunks/ py objects in the file.
    print('File saved successfully in .pkl format')

text = load_file()
chunks = split_text(text)
save_chunks_pickle(chunks)

#-----------------------------------------------------------------------#-----------------------------------------------------------#-----------------------------------------------------#-----------------

# Main Execution Flow

# Step 1: Load extracted text
text = load_file()

# Step 2: Create chunks
chunks = split_text(text)

# Optional:
view_chunks(chunks)

