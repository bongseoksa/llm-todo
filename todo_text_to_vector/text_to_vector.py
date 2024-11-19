from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from constants import long_text

# 1. 텍스트 분할
def split_text_into_documents(text, separator=" ", chunk_size=100, chunk_overlap=20):
    text_splitter = CharacterTextSplitter(separator=separator, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_texts = text_splitter.split_text(text)
    return [Document(page_content=text) for text in split_texts]

# 2. 벡터화
def embed_documents(documents, model_name="jhgan/ko-sbert-nli"):
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    return embedding_model.embed_documents([doc.page_content for doc in documents])

# 3. FAISS 인덱스 생성 및 벡터 추가
def create_faiss_index(documents, vectors, embedding_model):
    text_embeddings = list(zip([doc.page_content for doc in documents], vectors))
    return FAISS.from_embeddings(text_embeddings, embedding_model)

# 4. 질의 응답
def query_documents(db, query, k=3):
    docs_with_scores = db.similarity_search_with_score(query, k=k)
    return [(doc.page_content, score) for doc, score in docs_with_scores]

# 메인 실행
documents = split_text_into_documents(long_text)
vectors = embed_documents(documents)
db = create_faiss_index(documents, vectors, HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli"))
query = '우량 사업 중심'
top_docs_with_scores = query_documents(db, query, 5)
for doc, score in top_docs_with_scores:
    print(f"연관도 점수: {score}\n문서: {doc}\n")