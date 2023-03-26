# import packages
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain import OpenAI, VectorDBQA
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler



#function to load and process pdf ie. split into chunks
def load_process_pdf(filename):
    #Loading pdf
    loader = PyPDFLoader(f"{filename}")
    documents = loader.load()
    #documents is an array containing all the pages in the pdf
    #each index is a page (type document object that has the following properties:
    #page_content (actual content itself) and some meta data having info about the source and page number (here index no)

    #splitting pdf into chunks containing 1000 characters
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    #in this case texts is exactly the same as documents
    return texts

#run only ONCE
def generate_store_embeddings(OPENAI_API_KEY, PINECONE_API_ENV,PINECONE_API_KEY,index_name,texts):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    pinecone.init(
    api_key=PINECONE_API_KEY, 
    environment=PINECONE_API_ENV  
    )
    index_name = index_name
    Pinecone.from_documents(documents = texts, embedding = embeddings, index_name=index_name)


def fetch_embeddings(index_name, embeddings):
    db = Pinecone.from_existing_index(index_name = index_name, embedding= embeddings)
    return db


def vector_dbqa_chain_config(db, chain_type, OPENAI_API_KEY):
    llm = OpenAI(openai_api_key= OPENAI_API_KEY, temperature= 0,streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])) 
    qa = VectorDBQA.from_chain_type(llm = llm, chain_type = chain_type, vectorstore = db, verbose = True)
    return qa
