from utils import fetch_embeddings,vector_dbqa_chain_config
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import streamlit as st
from test import test_answer

#initializing embeddings and pinecone class
pinecone.init(
    api_key=st.secrets.PINECONE_API_KEY,
    environment=st.secrets.PINECONE_API_ENV
)
embeddings = OpenAIEmbeddings(openai_api_key=st.secrets.OPENAI_API_KEY)
index_name = "sparkmate-welcome-deck"
db = fetch_embeddings(index_name= index_name, embeddings= embeddings)

#setting up chains
qa_general = vector_dbqa_chain_config(db=db, chain_type= "stuff", OPENAI_API_KEY=st.secrets.OPENAI_API_KEY)


#Frontend

#importing css file
with open("design.css") as source_des:
   st.markdown(f"<style>{source_des.read()}</style>",  unsafe_allow_html=True)


st.title('🔥 I am Sparkmate GPT 🤖 ')
st.title(" 😉 Ask me anything")
search_query = st.text_input(label = " ", placeholder= "Explain the Sharelock project?")

st.text(" ") 

if search_query:
    st.text(" ")
    with st.spinner('Wait for it...'):
        answer = qa_general.run(search_query)
        st.title("💡 Answer")
        st.text(" ")
        st.write(f'<div style="color: #ef4775;"><span style="font-style: normal; font-size: 18px">{answer}</span></div>', unsafe_allow_html=True)
        st.text(" ")
        st.text(" ")
        st.markdown("ℹ️ Check out the original document [here](https://uploads-ssl.webflow.com/624abfd0aa25d61251e243e2/6282603961b9df74881ac0d0_sparkmate-deck-2022-.pdf)")
    
        
