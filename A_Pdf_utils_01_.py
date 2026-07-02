'''
Day - 01
We will extract all the text from our PDF
We will store entire text in one text file.
'''

import fitz
# Fitz also known as PyMuPDF powerful open-source Python library designed for working with PDF documents.

import os
# The os module in Python provides a way to interact with the operating system.
# It includes functions to handle file operations, directory management, and other OS-related tasks.


# STEP 1: EXTRACT THE PAGES:

def extract_text(pdf):
    # This function opens the PDF and extracts text from each page.

    if not os.path.exists(pdf):
        # If os does not find my PDF in the ROOT DIRECTORY then this code would be executed.
        print("PDF file does not exist.")
        return None

    doc = fitz.open(pdf)
    # fitz opens the PDF.

    print(f"Total Pages: {len(doc)}")

    # STEP 2: LOOP THROUGH EVERY PAGE:

    all_text = []
    # This variable stores all the text extracted from each page.

    for i in range(len(doc)):
        # Total_pages = 5, i --> 0, 1, 2, 3, 4

        page = doc[i]
        # doc[0] ----> Opens Page 1

        text = page.get_text()
        # .get_text() extracts text from the i-th page.

        all_text.append(text)
        # .append() stores the extracted text of each page into the list.

        print(f"Page {i + 1} extracted successfully.")
        # Displays extraction status for each page.

    doc.close()
    # Closes the PDF file after extraction.

    # STEP 3: JOIN ALL PAGES INTO A SINGLE STRING

    full_text = "\n".join(all_text)
    # Combines text from all pages into one string.

    return full_text


# STEP 4: SAVE TO extract_text.txt

def save_file(text):

    if text is None:
        print("No text available to save.")
        return

    with open('extract_text.txt', 'w', encoding='utf-8') as file:
        # 'w' is the write mode which creates the file if it does not exist.
        # encoding='utf-8' is used to handle Unicode characters properly.

        file.write(text)

        print("All the text from each page of PDF is extracted successfully.")


# 1. Extract_Text

text = extract_text('Intro_ML.pdf')

# 2. Save_File

save_file(text)