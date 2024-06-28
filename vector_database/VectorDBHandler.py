from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

class VectorDBHandler:
    def __init__(self):
        self.vector_db = None
        self.docs = None

    def load_and_process_data(self, dataset_name, page_content_column, chunk_size=1000, chunk_overlap=150):
        loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)
        data = loader.load()
        
        for doc in data:
            doc.page_content = doc.page_content.encode().decode('unicode_escape')
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.docs = text_splitter.split_documents(data)
        
        return self.docs

    def create_embeddings(self, model_path, device='cpu', normalize_embeddings=False):
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': normalize_embeddings}

        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return embeddings

    def create_vector_db(self, model_path, device='cpu', normalize_embeddings=False):
        if self.docs is None:
            raise ValueError("Documents are not loaded and processed yet.")
        
        embeddings = self.create_embeddings(model_path, device, normalize_embeddings)
        self.vector_db = FAISS.from_documents(self.docs, embeddings)
        return self.vector_db

    def add_documents_to_db(self, docs, model_path, device='cpu', normalize_embeddings=False):
        embeddings = self.create_embeddings(model_path, device, normalize_embeddings)
        if not self.vector_db:
            self.vector_db = FAISS.from_documents(docs, embeddings)
        else:
            self.vector_db.add_documents(docs)

    def get_vector_db(self):
        return self.vector_db
