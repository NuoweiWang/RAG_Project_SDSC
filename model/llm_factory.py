import os
from langchain_openai import ChatOpenAI

class LLMFactory:
    def __init__(self):
        self.llm_instances = {}

    def create_openai_llm(self, api_key, api_base, temperature=0.95, model="glm-4-0520"):
        llm = ChatOpenAI(
            temperature=temperature,
            model=model,
            openai_api_key=api_key,
            openai_api_base=api_base
        )
        self.llm_instances["openai"] = llm
        return llm

    def create_tongyi_llm(self, api_key, api_base, temperature=0.95, model="default_model"):
        llm = TongyiAPI(
            temperature=temperature,
            model=model,
            api_key=api_key,
            api_base=api_base
        )
        self.llm_instances["tongyi"] = llm
        return llm