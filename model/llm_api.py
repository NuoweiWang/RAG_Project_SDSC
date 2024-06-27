from .chat_zhipuai import ChatZhipuAI

class zhipu_llm:
    _instance = None
    
    @staticmethod
    def getInstance():
        if zhipu_llm._instance is None:
            zhipu_llm()
        return zhipu_llm._instance

    def __init__(self):
        if zhipu_llm._instance is not None:
            raise ValueError("An instantiation already exists!")
        else:
            zhipu_llm._instance = ChatZhipuAI(model_name="glm-4", api_key = '8adaf2c8b02bd171d7bd54c93f03f6b3.krO1SVk6CVP0DitS', temperature = 0.5)
