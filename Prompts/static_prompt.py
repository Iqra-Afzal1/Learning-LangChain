import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os

load_dotenv(override=True)

class ConfiguerModel:

    def __init__(self, model_name:str = "Qwen/Qwen2.5-7B-Instruct", task:str = "text-generation"):
        self.model_name = model_name
        self.model = None
        self.task = task
        self._initiate_model()

    def _initiate_model(self):
        print(f"Trying to connect to model {self.model_name}...")
        try:
            llm = HuggingFaceEndpoint(
                repo_id = self.model_name,
                task = self.task,
                huggingfacehub_api_token= os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')
            )
            self.model = ChatHuggingFace(llm=llm)
            print("Connected to model successfully !")

        except Exception as e:
            print(e)
            raise

    def get_response(self, prompt):
        print("Query sent to model, waiting for model model response...")
        try:
            response = self.model.invoke(prompt)
            print("Response received successfully !")
            return response.content
        
        except Exception as e:
            print(e)
            raise

# to avoid creation of object again and again as streamlit reruns complete script everytime a change is created on the page
@st.cache_resource
def get_object():
    model_configurer = ConfiguerModel()
    return model_configurer


st.title("Ask Anything")
prompt = st.text_input("Enter your query")

model_configuer = get_object()

# if user will click on button, we'll fetch response from model and show it to user
if st.button("Ask", key=1):
    response = model_configuer.get_response(prompt)
    st.write(response)
