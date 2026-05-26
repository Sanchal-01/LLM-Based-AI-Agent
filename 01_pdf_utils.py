'''
Day - 01 
We will extract all the text from our PDF
We will store entire text in on text file.
'''

import fitz
import os


def extract_text(pdf): # This function opens pdf and will extract text from each pages.
    pass

    if not os.path.exists(pdf):  # Agar mera pdf folder mein exist nahi karta hai to:
        print("Folder does not exist.")

    
    doc = fitz.open(pdf)   # fitz open karo pdf 
    # print(len(doc))


# STEP 2: Loop through every Page 
    all_text = []     # ye ek variable hain jiske ander saara text aa jayega jo har ith iteration ke baad extract hoga.

    for i in range(len(doc)):  # Total_page = 5, i -->0, 1, 2, 3, 4, 5   
        page = doc[i]         # doc[0]----> Page 1 open karega
        text = page.get_text()     # .get_text ---> ith page se text extract karega. 
        all_text.append(text)      # .append se hum pichle wale i th value ke text ko append karte jayenge ek sath ek hi text file mein.

    # print(all_text)     # This introduces \n we don't want this.


    # Step 4: Join all pages without \n between text.
    full_text = "\n".join(all_text)
    return(full_text)
    
extract_text('sample.pdf')



# Step 5 : Save to extracted_text.txt

def save_file(text):
    with open('extract_text.txt', 'w') as file :     # w is the mode which creates file.  # file is the variable in wich the values are stored.
        file.write(text)
        print("All the text from each page of PDF is extracted successfully")



#1. Extract_Text 
text = extract_text('sample.pdf')

#2. Save_File  
save_file(text)