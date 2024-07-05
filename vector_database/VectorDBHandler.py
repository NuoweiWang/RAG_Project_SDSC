from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

class VectorDB:
    def __init__(self):
        self.data = None
        self.docs = None
        self.embeddings = None
        self.db = None

    def load_data(self, dataset_name, page_content_column):
        # 加载数据集
        loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)
        self.data = loader.load()
        # 解决编码问题
        for doc in self.data:
            doc.page_content = doc.page_content.encode().decode('unicode_escape')

    def split_data(self, chunk_size=1000, chunk_overlap=150):
        # 分割数据集为文档
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.docs = text_splitter.split_documents(self.data)

    def create_embeddings(self, model_path, device='cuda', normalize_embeddings=False):
        # 创建嵌入向量
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': normalize_embeddings}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    def load_to_faiss(self, save_path):
        # 确保已创建嵌入向量
        if not self.embeddings:
            raise ValueError("请先创建嵌入向量。")
        # 将文档加载到FAISS并保存
        self.db = FAISS.from_documents(self.docs, self.embeddings)
        self.db.save_local(save_path)

    def load_from_faiss(self, load_path):
        # 确保已创建嵌入向量
        if not self.embeddings:
            raise ValueError("请先创建嵌入向量。")
        self.db = FAISS.load_local(load_path, self.embeddings)

    def similarity_search(self, question):
        # 进行相似度搜索
        search_docs = self.db.similarity_search(question)
        text = []
        for i, doc in enumerate(search_docs):
            text.append(f"information{i+1}: {doc.page_content}\n")
        return " ".join(text)
