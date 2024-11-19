from langchain_openai import ChatOpenAI
import os
import time

from openai import OpenAI

# client = OpenAI(
#     organization='org-13exAkRST1KtvEH5NF607Uaf',
#     project='$PROJECT_ID',
# )

# os.environ['OPENAI_API_KEY'] = ''

# # model 
# llm = ChatOpenAI(model="gpt-3.5-turbo")

# # chain 실행
# try:
#     llm.invoke("지구의 자전 주기는?")
# except client.error.RateLimitError:
#     print('할당량 초과.')
#     time.sleep(60)

import getpass

os.environ["OPENAI_API_KEY"] = getpass.getpass()

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)