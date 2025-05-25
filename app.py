from typing import List
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from neo4j import TRUST_ALL_CERTIFICATES, GraphDatabase
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from utils import retriever
import ssl
print(ssl.OPENSSL_VERSION)

load_dotenv()

app = FastAPI()

print(f"NEO4J_URI: {os.getenv('NEO4J_URI')}")
print(f"NEO4J_PASSWORD: {os.getenv('NEO4J_PASSWORD')}")
print(f"NEO4J_USERNAME: {os.getenv('NEO4J_USERNAME')}")

# Serve React app (build)
client_path = Path("client/out")
app.mount("/static", StaticFiles(directory=client_path), name="static")

class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities  that " "appear in the text",
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:

        while True:
            data = await websocket.receive_json()
            message=data["text"]
            print(f"Received message: {message}")

            graph = Neo4jGraph(
                url="neo4j+s://5e4610f3.databases.neo4j.io",
                username="neo4j",
                password="ZRBjoQsMUFKVm70MxggvCUZEYlJBaDhjM-96Fv7MA3k",
                refresh_schema=True)

            llm = AzureChatOpenAI(
                azure_deployment="gpt-4.1",
                openai_api_version="2024-12-01-preview",
                temperature=0.4
            )

            embeddings_3_large : AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
                azure_deployment="text-embedding-3-large",
                openai_api_version="2024-12-01-preview",
                dimensions=3072
            )

            vector_index = Neo4jVector.from_existing_graph(
                embedding=embeddings_3_large,
                search_type="hybrid",
                node_label="Document",
                text_node_properties=["text"],
                embedding_node_property="embedding"
            )

            prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You are extracting organization and person entities from the text.",
                        ),
                        (
                            "human",
                            "Use the given format to extract information from the following"
                            "input: {question}",
                        ),
                    ]
                )
            
            entity_chain = prompt | llm.with_structured_output(Entities)

            template = """
                You are a virtual assistant tasked with answering questions from the user about a specific candidate.

                You will be provided with the following information:

                - The user’s question
                - A context consisting of:
	                1. Structured Data derived from a knowledge graph
	                2. Unstructured Data extracted from documents stored in a vector database, all related to the candidate in question
               
                Your goal is to analyze the provided context and craft a final response that highlights the strengths of the candidate.

                How to respond:
	            - Use only the information provided in the context.
	            - Do not mention the sources or explain how you derived the information. Just answer the question as if you already know the candidate.
	            - If the question is irrelevant or cannot be answered based on the context, reply with: “I’m sorry, but I can’t answer that question.”
	            - Structure your response clearly:
		            - Open with a short summary or direct answer
	                - Use paragraphs to separate different ideas
	                - Highlight important skills, achievements, or traits in bold
	                - Use Markdown formatting for clarity and readability
	            - Rephrase and summarize the information—don’t copy-paste, and don’t invent.
	            - Your answer should sound natural and professional, as if you’re helping the user get a quick but insightful picture of the candidate.
	            - The final response must be clear, polished, and make the candidate's strengths and experiences easy to understand and appreciate.

                User Input: {question}
                
                Context: {context}

                Respond in English.
                """

            prompt = ChatPromptTemplate.from_template(template)

            chain = (
                RunnableParallel(
                    {
                        "context": RunnableLambda(lambda x: retriever(graph, vector_index,x["question"], entity_chain )),
                        "question": RunnablePassthrough(),
                    }
                )
                | prompt
                | llm
                | StrOutputParser()
            )

            result = chain.invoke({"question": message})
            await websocket.send_json({"role":"ai", "content":result})

    except WebSocketDisconnect:
        print("errore")

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    target_path = client_path / full_path
    if target_path.is_dir():
        index = target_path / "index.html"
        if index.exists():
            return FileResponse(index)
    elif target_path.exists():
        return FileResponse(target_path)

    fallback_index = client_path / "index.html"
    return FileResponse(fallback_index)
