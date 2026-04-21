import streamlit as st
from langserve import RemoteRunnable


def get_groq_response(input_text, language):
    chain = RemoteRunnable("http://127.0.0.1:8000/chain")
    response = chain.invoke({"language": language, "text": input_text})
    print(response)
    return response


## Streamlit app
st.title("LLM Application Using LCEL")

language = st.selectbox(
    "Select target language",
    ["French", "Spanish", "German", "Italian", "Portuguese", "Hindi", "Japanese", "Chinese", "Arabic", "Russian"]
)

input_text = st.text_input("Enter the text you want to translate")

if input_text:
    st.write(get_groq_response(input_text, language))
