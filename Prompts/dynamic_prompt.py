import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv(override=True)


class ConfigureModel:
    def __init__(self, model_name:str = "Qwen/Qwen2.5-7B-Instruct", task:str = "text-generation"):
        self.model_name = model_name
        self.task = task
        self.model = None
        self._initiate_model()

    def _initiate_model(self):
        print(f"Trying to connect with model {self.model_name}...")
        try:
            llm = HuggingFaceEndpoint(
                repo_id = self.model_name,
                task = self.task,
                huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')
                )
            self.model = ChatHuggingFace(llm = llm)
            print("Connected to model successfully !")

        except Exception as e:
            print(e)
            raise

    def get_response(self, prompt:PromptTemplate):
        print("Query sent to model, waiting for response...")
        try:
            response = self.model.invoke(prompt)
            print("Response received successfully !")

            return response.content
        
        except Exception as e:
            print(e)
            raise
            
@st.cache_resource
def get_object():
    model_configurer = ConfigureModel()
    return model_configurer


st.title("Ask anything about medicine")
user_prompt = st.text_input("Write your query here")

# Creating a dynamic prompt 
template = PromptTemplate(
    template = """
    You are a MBBS doctor with highly qualified skills. Answer {user_prompt} in less than 5 lines
    """,
    input_variables=['user_prompt']
)
prompt = template.invoke({
    'user_prompt' : user_prompt
})

model_configurer = get_object()

if st.button("Ask"):
    response = model_configurer.get_response(prompt)
    st.write(response)
