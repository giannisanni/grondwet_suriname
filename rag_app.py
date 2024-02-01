# from dotenv import load_dotenv
# import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
def main():
    # load_dotenv()
    st.set_page_config(page_title="Grondwet Suriname")
    #st.header("Grondwet Suriname")
    st.markdown("<h1 style='text-align: center;'>Grondwet Suriname</h1>", unsafe_allow_html=True)

    # upload file
    #pdf = st.file_uploader("your pdf", type="pdf")
    pdf = "grondwet_suriname_no_sources.pdf"

    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # show user input
        user_question = st.chat_input("Ask a question about the constitution of Suriname:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)
            st.info(f"💬: {user_question}")
            st.success(f"🧠: {response}")


if __name__ == '__main__':
    main()

