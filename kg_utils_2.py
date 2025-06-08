import base64
import json
from typing import List
import fitz
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph, Neo4jVector
import asyncio
from langchain_core.documents import Document
from langchain_neo4j import GraphCypherQAChain
import os
from io import BytesIO
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter, MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from markitdown import MarkItDown
from openai import AzureOpenAI
import pdfplumber

# FILES_FOLDER = "files/Simone Bitti/Simone Bitti CV.pdf"
FILES_FOLDER = "files/Simone Bitti/Simone Bitti - Cover Letter.pdf"


def summarize_document(html_pages):
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4.1",
        openai_api_version="2024-12-01-preview",
        temperature=0.0
    )

    text=str(html_pages)
   
    template=f"""
    You will be provided with an HTML document containing information from a résumé (CV) or  cover letter higlightinggoogle googl.

    Your task is to:
    - **Accurately extract** all relevant content from the HTML document.
    - Write a **highly detailed summary** that includes every useful piece of information from the document.
    - Improve the clarity of the summary by rephrasing experiences, skills, and technologies to make them easier to understand, if necessary.
    - **Do not** add any information, experiences, or technologies that are not explicitly mentioned in the original document.

    Important guidelines for the summary:
    - Do not omit anything: every relevant detail must be included.
    - Do not add any personal opinions or interpretations.
    - The summary should be **comprehensive** and **well-structured**.
    - The summary should be **in English**.

    Return **only** the text of the summary, with no additional comments or formatting.

    Here is the HTML document to analyze: {text}
    """
    
    ai_msg = llm.invoke([HumanMessage(content=template)])
    response  = ai_msg.content
    
    try:
        return  response
    except json.JSONDecodeError as e:
        print("Failed to decode JSON:", e)
        return None

def convert_img_to_html(encoded_image):

    system_prompt = """You will be provided with an image representing a page from a PDF, which may contain text, images, tables, and other information.

    Your task is to carefully analyze the image and extract **all the information** in **HTML format**.

    Follow these instructions:
    - If the image is empty or does not contain any useful information, produce no output.
    - Do **not** add any colors or styles to the generated HTML.
    - The generated HTML must **faithfully preserve the structure** represented in the image, maintaining the importance and order of all extracted information.

    Your output should be **only** the generated HTML based on the input image.
    """

    prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_prompt),
                HumanMessage(content=[
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                    }
                ])
            ])

   
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4.1",
        openai_api_version="2024-12-01-preview",
        temperature=0.0
    )

    chain = prompt | llm

    response = chain.invoke({})

    html = response.content
    
    return html

async def create_kg():

    # endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    # deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4.1")
    # subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

    # # Inizializzare il client OpenAI di Azure con l'autenticazione basata su chiave
    # client = AzureOpenAI(
    #     azure_endpoint=endpoint,
    #     api_key=subscription_key,
    #     api_version="2025-01-01-preview"
    # )

    # md = MarkItDown(llm_client=client, llm_model=deployment)
    # result = md.convert(FILES_FOLDER)
    # print(result.text_content)


    with pdfplumber.open(FILES_FOLDER) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"

    print(text)

    llm = AzureChatOpenAI(
        azure_deployment="gpt-4.1",
        openai_api_version="2024-12-01-preview",
        temperature=0.0
    )

    prompt = f"""Questo è il testo di un CV estratto da PDF.

    Organizza tutto il testo in sezioni (Nome e Cognome, Ruolo, Skills, Profilo, Esperienze, Formazione, Contatti, Lingue, ecc.) usando solo titoli e sottotitoli markdown dove serve.

    IMPORTANTE:
    - Non modificare, riscrivere, riassumere o omettere nulla del testo.
    - Non inventare nulla.
    - Non cambiare le frasi: tutto il contenuto del testo deve essere incluso integralmente, solo diviso in sezioni.
    - Non aggiungere grassetto, corsivo, tabelle o stili.
    - Se hai dubbi su dove inserire una parte, mantienila nella sezione che sembra più corretta senza modificare il testo originale.

    TESTO PDF:

    {text}
    """

    ai_msg = llm.invoke([SystemMessage(content="Sei un assistente esperto di curriculum, conver letter e markup."),HumanMessage(content=prompt)])
    response  = ai_msg.content

    print(response)
    
    graph = Neo4jGraph(refresh_schema=False)

    # graph.query("""MATCH (n) DETACH DELETE n""")
    # print("Graph cleared.")

    embeddings_3_large : AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-3-large",
        openai_api_version="2024-12-01-preview",
        dimensions=3072
    )

    # text_splitter = SemanticChunker(embeddings_3_large)
    # documents = text_splitter.create_documents([response], [{"source": FILES_FOLDER}])


    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    documents = markdown_splitter.split_text(response)

    for doc in documents:
        doc.metadata = {"source": FILES_FOLDER}
        doc.page_content = f"This document is a chunk of the original document {doc.metadata['source']} and refers to Simone Bitti.\n\n {doc.page_content}."
  
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4.1",
        openai_api_version="2024-12-01-preview",
        temperature=0.0
    )

    llm_transformer = LLMGraphTransformer(llm=llm)

    graph_documents = await llm_transformer.aconvert_to_graph_documents(documents)
    print(f"Nodes:{graph_documents[0].nodes}")
    print(f"Relationships:{graph_documents[0].relationships}")

    graph.add_graph_documents(graph_documents, include_source=True, baseEntityLabel=False)      
    graph.close()

if __name__ == "__main__":
    asyncio.run(create_kg())
    