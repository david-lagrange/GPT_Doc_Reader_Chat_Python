import os
import openai
import re
from time import time
from dotenv import load_dotenv
import pinecone
from uuid import uuid4
import datetime
import langchain
import textract
import PyPDF2
from ebooklib import epub
import nltk
import pymongo
from uuid import uuid4
import time

# nltk.download('punkt')
load_dotenv()

prompt_template = '''I am a chatbot name DOC_BOT. My goals are to reduce suffering, increase prosperity, and increase understanding. 
I will read the context notes, recent messages, and then I will provide a long, verbose, detailed answer. I will use any numbers I find
in the context information if applicable to my message, and I will NOT make up any numbers.

Contextual Information:
<<CONTEXT>>

USER: <<MESSAGE>>

I will now provide a long, detailed, verbose response, follwed by a question:
DOC_BOT:'''

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def average_chars_per_sentence(text):
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)

    # Calculate the total number of characters
    total_chars = sum(len(sentence) for sentence in sentences)

    # Calculate the average number of characters per sentence
    average_chars = total_chars / len(sentences)

    return average_chars

def read_file(filepath):
    _, file_extension = os.path.splitext(filepath)
    content = ""

    if file_extension.lower() in [".txt", ".doc", ".docx", ".rtf"]:
        content = textract.process(filepath).decode("utf-8")
    elif file_extension.lower() == ".pdf":
        with open(filepath, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            content = " ".join(
                [reader.pages[i].extract_text() for i in range(len(reader.pages))]
            )
    elif file_extension.lower() == ".epub":
        book = epub.read_epub(filepath)
        content = " ".join([item.content.decode("utf-8") for item in book.get_items() if item.get_type() == 9])

    return content


def chunk_text(content, chunk_size=100, chunk_overlap=20):
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size should be greater than chunk_overlap")

    chunks = []
    content_length = len(content)

    for start in range(0, content_length, chunk_size - chunk_overlap):
        end = min(start + chunk_size, content_length)
        chunk = content[start:end]
        chunks.append(chunk)

    return chunks

def store_data_in_mongo(collection, data):

    result = collection.insert_many(data)


def get_gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII', errors='ignore').decode()
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']
    return vector

def split_list(input_list, max_sublist_size=250):
    output_list = []
    for i in range(0, len(input_list), max_sublist_size):
        sublist = input_list[i:i + max_sublist_size]
        output_list.append(sublist)
    return output_list

def process_data_for_chat(collection, index, i_namespace, specific_filepath):
    raw_file_text = read_file(specific_filepath)

    # average_chars_per_sent = average_chars_per_sentence(raw_file_text)

    text_chunks = chunk_text(raw_file_text)

    split_text_chunks = split_list(text_chunks)

    total_text_strings = sum([len(sub_list) for sub_list in split_text_chunks])

    # data_objects = [{'id': str(uuid4()),'data': chunk} for chunk in text_chunks]
    # store_data_in_mongo(mongo_client, data_to_save_in_standard_db)

    total_count = 0
    for single_chunk_list in split_text_chunks:

        data_to_save_in_standard_db = []
        data_to_save_in_vector_db = []

        for single_text in single_chunk_list:
            total_count += 1
            print(f'{total_count}/{total_text_strings}')
            
            unique_id = str(uuid4())
            
            vector = get_gpt3_embedding(single_text)
            
            data_to_save_in_standard_db.append({"data_id": unique_id, 'data': single_text})
            data_to_save_in_vector_db.append((unique_id, vector))
            
        store_data_in_mongo(collection, data_to_save_in_standard_db)
        index.upsert(vectors=data_to_save_in_vector_db, namespace=i_namespace)
        time.sleep(10)

def gpt3_completion(prompt, engine='text-davinci-003', temp=0.0, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0, stop=['USER:', 'DOC_BOT:']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII', errors='ignore').decode()

    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop
            )
            text = response['choices'][0]['text'].strip()
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t]+', ' ', text)
            return text
        except Exception as err:
            retry += 1
            if retry >= max_retry:
                return "GPT3 ERROR: %s" % err
                print('Error communicating with OpenAI:', err)

def chat_with_document(collection, index):

    number_of_relative_results = 20

    while True:
        # Get user input and conver to vector
        user_question = input('\n\nUSER: ')
        vector = get_gpt3_embedding(user_question)

        # Query for similar vector results
        results = index.query(vector=vector, top_k=number_of_relative_results, include_values=False, include_metadata=False)
        context_ids = [result['id'] for result in results['matches']]

        # Query for raw data connected to id
        raw_datas = collection.find({"data_id": {"$in": context_ids}})
        context = '\n'.join([data['data'] for data in raw_datas])

        # Construct prompt
        prompt = prompt_template.replace('<<CONTEXT>>', context).replace('<<MESSAGE>>', user_question)

        # Ask GPT
        output = gpt3_completion(prompt)

        # Display
        print(f'\n\n{output}')

if __name__ == '__main__':
    openai.api_key = os.getenv('OPENAI_API_KEY')
    pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENVIRONMENT'))
    mongo_password = os.getenv('MONGO_DB_PASSWORD')
    mongo_client = pymongo.MongoClient(f'mongodb+srv://dlagrange:{mongo_password}@cluster0.stcb9nj.mongodb.net/?retryWrites=true&w=majority')
    index = pinecone.Index('gpt-reader-index')
    i_namespace = 'doc-namespace'
    db = mongo_client["gpt-reader-db"]
    collection = db["data-store"]

    should_process_input = True if input('Process new data? Y\\n ') == 'Y' else False
    if should_process_input:
        # Clear Pinecone index and Mongo DB
        index.delete(deleteAll='true', namespace='i_namespace')
        collection.delete_many({})
        specific_filepath = './the_present.pdf'
        process_data_for_chat(collection, index, i_namespace, specific_filepath)
        print("Data has been processed. You can begin chatting with the document now.")


        
    # Begin chat bot
    chat_with_document(collection, index)

    pass