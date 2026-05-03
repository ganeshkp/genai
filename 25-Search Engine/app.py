import os

import streamlit as st
from dotenv import load_dotenv
from langchain_classic.agents import AgentType, initialize_agent
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.tools import (
    ArxivQueryRun,
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
)
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_groq import ChatGroq

## Arxiv and wikipedia Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper, handle_tool_error=True)

search = DuckDuckGoSearchRun(name="Search")


st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

## Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assisstant",
            "content": "Hi,I'm a chatbot who can search the web. How can I help you?",
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

CONVERSATIONAL_KEYWORDS = {"hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye", "ok", "okay", "cool", "great", "sure", "yes", "no", "yep", "nope"}

def is_conversational(text: str) -> bool:
    return text.strip().lower().rstrip("!.,?") in CONVERSATIONAL_KEYWORDS

if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(
        groq_api_key=api_key, model_name="llama-3.1-8b-instant", streaming=True
    )

    with st.chat_message("assistant"):
        if is_conversational(prompt):
            response = llm.invoke(prompt).content
            st.write(response)
        else:
            tools = [search, arxiv, wiki]
            search_agent = initialize_agent(
                tools,
                llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
            )
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            try:
                response = search_agent.run(prompt, callbacks=[st_cb])
            except Exception as e:
                response = f"Sorry, I encountered an error while searching: {type(e).__name__}. Please try again."
            st.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
