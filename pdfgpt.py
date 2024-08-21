import streamlit as st
import yaml
from loguru import logger
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import loading
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PDFPlumberLoader


def main():
    st.set_page_config(
    page_title="PDF챗봇",
    page_icon=":books:")

    st.title("_Private PDF 데이터 :red[QA 챗봇]_ :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        selected_model = st.selectbox(
        "LLM 선택", ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"], index=0
        )
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
                
        uploaded_files =  st.file_uploader("Upload your file",type=['pdf'],accept_multiple_files=True)
        process = st.button("Process")
        clear_btn = st.button("대화 초기화")


    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        if not selected_model:
            st.info("Please add your selected_model to continue.")
            st.stop()
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks, openai_api_key)
     
        st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key, selected_model) 

        st.session_state.processComplete = True

    if 'messages' not in st.session_state or clear_btn:
        st.session_state.messages = [{"role": "assistant", 
                                        "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("답변중입니다..."):
                result = chain({"question": query, "chat_history": st.session_state.chat_history})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']

                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
   
        st.session_state.messages.append({"role": "assistant", "content": response})


def load_prompt(file_path, encoding="utf8"):

    with open(file_path, "r", encoding=encoding) as f:
        config = yaml.safe_load(f)

    return loading.load_prompt_from_config(config)

def get_text(docs):
    doc_list = []
    
    for doc in docs:
        file_name = doc.name
        with open(file_name, "wb") as file:
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        # 일단 pdf 확장자만 가능하게 처리
        if '.pdf' in doc.name:
            loader = PDFPlumberLoader(file_name)
            documents = loader.load()

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks, openai_api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore, openai_api_key, selected_model):
    prompt = load_prompt("prompts/pdf-rag.yaml")
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = selected_model, temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = False,
            combine_docs_chain_kwargs={"prompt": prompt},
        )

    return conversation_chain


if __name__ == '__main__':
    main()