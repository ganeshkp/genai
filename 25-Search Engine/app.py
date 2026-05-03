import os

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_community.tools import (
    ArxivQueryRun,
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
)
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper, handle_tool_error=True)

search = DuckDuckGoSearchRun(name="Search")


def run_tool_safely(tool_name: str, query: str, runner) -> str:
    query = query.strip()
    if not query:
        return f"{tool_name} needs a non-empty search query."

    try:
        result = runner.invoke(query)
    except Exception as error:
        return (
            f"{tool_name} is temporarily unavailable for this query "
            f"({type(error).__name__}: {error}). Try another tool or answer from "
            "general knowledge if enough context is available."
        )

    if not result:
        return f"{tool_name} did not return results for: {query}"

    return str(result)


@tool
def web_search(query: str) -> str:
    """Search the web for recent or general information."""
    return run_tool_safely("Web search", query, search)


@tool
def arxiv_search(query: str) -> str:
    """Search arXiv for research papers and scientific articles."""
    result = run_tool_safely("arXiv search", query, arxiv)
    if "is temporarily unavailable" not in result:
        return result

    fallback_query = f"{query} paper"
    fallback = run_tool_safely("Web search fallback", fallback_query, search)
    return f"{result}\n\nFallback web result:\n{fallback}"


@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for encyclopedic background information."""
    return run_tool_safely("Wikipedia search", query, wiki)


tools = [web_search, arxiv_search, wikipedia_search]

st.title("🔎 LangChain - Chat with search")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input(
    "Enter your Groq API Key:",
    type="password",
    help="Leave blank to use GROQ_API_KEY from your .env or shell environment.",
)
if not api_key and not GROQ_API_KEY:
    st.sidebar.warning("Set GROQ_API_KEY in .env or paste your key here.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi, I'm a chatbot who can search the web. How can I help you?",
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

CONVERSATIONAL_KEYWORDS = {
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "bye",
    "goodbye",
    "ok",
    "okay",
    "cool",
    "great",
    "sure",
    "yes",
    "no",
    "yep",
    "nope",
}

SEARCH_KEYWORDS = {
    "arxiv",
    "citation",
    "current",
    "latest",
    "news",
    "paper",
    "papers",
    "recent",
    "research",
    "search",
    "source",
    "sources",
    "today",
    "web",
    "wikipedia",
}


def is_conversational(text: str) -> bool:
    return text.strip().lower().rstrip("!.,?") in CONVERSATIONAL_KEYWORDS


def needs_search(text: str) -> bool:
    normalized = text.lower()
    return any(keyword in normalized for keyword in SEARCH_KEYWORDS)


def build_messages(messages: list) -> list:
    result = []
    for msg in messages:
        if msg["role"] == "user":
            result.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            result.append(AIMessage(content=msg["content"]))
    return result


def message_text(message) -> str:
    if isinstance(message.content, str):
        return message.content

    return "".join(
        block.get("text", "")
        for block in message.content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def user_facing_error(error: Exception) -> str:
    detail = str(error).strip()
    if detail:
        return f"Sorry, I encountered an error: {type(error).__name__}: {detail}"

    return f"Sorry, I encountered an error: {type(error).__name__}. Please try again."


if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    groq_api_key = api_key or GROQ_API_KEY
    if not groq_api_key:
        response = "Please add your Groq API key in the sidebar or set GROQ_API_KEY in your .env file."
        st.chat_message("assistant").write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.stop()

    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama-3.1-8b-instant",
        streaming=True,
        temperature=0,
    )

    with st.chat_message("assistant"):
        if is_conversational(prompt) or not needs_search(prompt):
            try:
                response = message_text(
                    llm.invoke(build_messages(st.session_state.messages))
                )
                st.write(response)
            except Exception as e:
                response = user_facing_error(e)
                st.error(response)
        else:
            agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt=(
                    "You are a helpful search assistant. Use the available tools "
                    "for current events, web search, Wikipedia, and arXiv questions. "
                    "Cite the tool output briefly when it supports your answer."
                ),
            )
            history = build_messages(st.session_state.messages)

            response_placeholder = st.empty()
            response = ""
            try:
                with st.status("Thinking...", expanded=True) as status:
                    for chunk in agent.stream(
                        {"messages": history},
                        stream_mode="updates",
                    ):
                        if "tools" in chunk:
                            for tool_msg in chunk["tools"]["messages"]:
                                st.write(
                                    f"**Tool:** `{tool_msg.name}` → {tool_msg.content[:200]}"
                                )
                        if "model" in chunk:
                            ai_msg = chunk["model"]["messages"][-1]
                            text = message_text(ai_msg)
                            if text:
                                response = text
                                response_placeholder.write(response)
                    if not response:
                        response = message_text(llm.invoke(history))
                        response_placeholder.write(response)
                    status.update(label="Done", state="complete", expanded=False)
            except Exception as e:
                try:
                    response = message_text(llm.invoke(history))
                    st.write(response)
                except Exception:
                    response = user_facing_error(e)
                    st.error(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
