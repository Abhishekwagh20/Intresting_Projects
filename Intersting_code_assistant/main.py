# main.py
import streamlit as st
from memory_manager import ProjectMemory
from api_handler import DeepseekAPI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class CodeAssistant:
    def __init__(self):
        self.memory = ProjectMemory()
        self.api = DeepseekAPI()
        
    def initialize_session(self):
        if 'project_root' not in st.session_state:
            st.session_state.project_root = None
        if 'web_search' not in st.session_state:
            st.session_state.web_search = True
        if 'r1_reasoning' not in st.session_state:
            st.session_state.r1_reasoning = False

    def render_sidebar(self):
        with st.sidebar:
            st.title("Configuration")
            st.session_state.web_search = st.checkbox("Enable Web Search", value=True)
            st.session_state.r1_reasoning = st.checkbox("Enable R1 Reasoning", value=False)
            
            if st.button("Load Project"):
                project_path = st.text_input("Enter project root path")
                if project_path:
                    self.memory.load_project(project_path)
                    st.session_state.project_root = project_path

    def chat_interface(self):
        st.title("Code Assistant")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        if prompt := st.chat_input("Ask about your project"):
            context = self.memory.get_relevant_context(prompt)
            response = self.api.query(
                prompt=prompt,
                context=context,
                web_search=st.session_state.web_search,
                r1_reasoning=st.session_state.r1_reasoning
            )
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                st.markdown(response)

    def run(self):
        self.initialize_session()
        self.render_sidebar()
        self.chat_interface()

if __name__ == "__main__":
    assistant = CodeAssistant()
    assistant.run()