import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
# from langchain.llms import OpenAI
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os


with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown(''' 
                ## About 
                Introducing our LLM-powered chatbot designed for seamless interaction with PDF documents. Whether you need to ask questions, extract data, or discuss content, our chatbot is your go-to companion. With advanced natural language processing, it offers efficient and accurate responses, revolutionizing how users engage with PDFs. Join us in embracing the future of document-based conversational AI.
                ''')
    st.write('Made with ❤️ by Denis')


 
def main():
    load_dotenv()
    st.header("Chat with your PDF 💬")
    
    # Initialize session state for chat history
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF here", type='pdf')
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # embeddings storing setup
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
    
        if os.path.exists(f"D:\ChatWithPDF\{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
                st.write('Embeddings Loaded from the Disk')
        else:
            # embeddings = OpenAIEmbeddings()
            embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
 
        
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
    
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
 
            llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0.5, "max_length":512})
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)

            # Update the chat history
            st.session_state.history.append({"query":query, "response":response})    
            st.write(response)

            # Display chat history
            st.markdown("### Chat History")
            for chat in st.session_state.history:
                st.markdown(f"**Q:** {chat['query']}")
                st.markdown(f"**A:** {chat['response']}")
                st.markdown("---")
 
if __name__ == '__main__':
    main()

