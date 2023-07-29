from dataclasses import dataclass
from enum import Enum

import streamlit as st
from langchain import LLMChain, PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)

from langchain_stuff.utils import project_path


class Role(str, Enum):
    USER = "user"
    BOT = "assistant"


@dataclass
class ChatMessage:
    role: Role
    content: str


@st.cache_resource
def get_llm_chain():
    # Load model
    cache_dir = project_path / "models"
    model_id = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, load_in_8bit=True, cache_dir=cache_dir
    )

    # Create initial prompt
    template = """Question: {question}

    Answer: Let's think step by step."""
    prompt = PromptTemplate.from_template(template)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=100,
    )

    # Create chain
    local_llm = HuggingFacePipeline(pipeline=pipe)
    llm_chain = LLMChain(prompt=prompt, llm=local_llm)

    return llm_chain


def main():
    st.write("# Langchain Streamlit app")

    if "chain" not in st.session_state:
        st.session_state["chain"] = get_llm_chain()

    if "conversation" not in st.session_state:
        st.session_state["conversation"] = []

    llm_chain = st.session_state["chain"]
    conversation = st.session_state["conversation"]

    prompt = st.chat_input("User: ", key="prompt")
    if prompt:
        conversation.append(ChatMessage(role=Role.USER, content=prompt))
        response = llm_chain(prompt)
        conversation.append(ChatMessage(role=Role.BOT, content=response["text"]))

    for message in conversation:
        with st.chat_message(message.role.value):
            st.write(message.content)

    st.write()


if __name__ == "__main__":
    main()
