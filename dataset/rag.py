
import json
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai import VertexAI
from langchain_community.vectorstores import Chroma
from vertexai.language_models import TextEmbeddingModel, TextGenerationModel
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
import vertexai

with open("tomato_diseases.json", "r") as f:  # Replace with your actual filename
    json_data = json.load(f)

documents = []  # List to store Langchain documents

for disease, treatment in json_data.items():
    document = Document(
        page_content=f"{disease}: {treatment}"
    )  # Combine disease and treatment
    documents.append(document)



PROJECT_ID = "qwiklabs-asl-03-66fd43168cb6"
LOCATION = "us-central1"

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)


embedding_function = VertexAIEmbeddings(
    model_name="text-embedding-004"
)
db = Chroma.from_documents(documents, embedding_function)

# query = "How to handle Bacterial spot?"
# docs = db.similarity_search(query, k=1)
# print(docs[0].page_content)

def ask_question(question: str):
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )


    llm = VertexAI(model_name="text-bison@001", max_output_tokens=1024)

    # q/a chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    prompt_template = f"""
    You are an expert in agricultural diseases. Please provide the recommendation for the following question:
    
    {question}
    
    Your response should be natural and conversational.
    """
    response = qa.invoke({"query":  prompt_template})
    return response['result']
    # print(f"Response: {response['result']}\n")