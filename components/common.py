import streamlit as st
import time
import threading


def github_logo():
    st.markdown(
        """
            <a href="https://github.com/your-github-repo">
                <svg height="32" aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="32" data-view-component="true" class="octicon octicon-mark-github v-align-middle" style="float:right;width:30px;height:30px;">
                <path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"></path>
            </svg>
            </a>
        """,
        unsafe_allow_html=True
    )


def api_token():
    st.sidebar.write("## 🔑 OpenAI API Token")
    api_token = st.session_state.get("api_token")
    if not api_token:
        api_token = "sk-"
    api_token_input = st.sidebar.text_input(
        "Enter OpenAI API Token", value=api_token)
    if api_token_input:
        st.session_state["api_token"] = api_token_input


def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"]::before {
                content: "OpenAI 102";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
            footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True
    )



def cost_calc():
    if "cost" not in st.session_state:
        st.session_state['cost'] = "$0.00"
    st.sidebar.write("## 💰 Cost Calculator")

    st.sidebar.markdown(
        "<sup>This is approx based on current page.</sup>", unsafe_allow_html=True)
    st.sidebar.markdown(
        f"**Estimated**: {st.session_state['cost']}")


def sidebar():
    add_logo()
    api_token()
    github_logo()
    cost_calc()
