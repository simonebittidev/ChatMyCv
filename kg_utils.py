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
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

FILES_FOLDER = "files/Simone Bitti"

def summarize_document(html_pages):
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4.1",
        openai_api_version="2024-12-01-preview",
        temperature=0.0
    )

    text=str(html_pages)
   
    template=f"""
    You will be provided with an HTML document containing information from a résumé (CV).

    Your task is to:
    - **Accurately extract** all relevant content from the HTML document.
    - Write a **highly detailed summary** that includes every useful piece of information from the document.
    - Improve the clarity of the summary by rephrasing experiences, skills, and technologies to make them easier to understand, if necessary.
    - **Do not** add any information, experiences, or technologies that are not explicitly mentioned in the original document.

    Guidelines:
    - Do not omit anything: every relevant detail must be included.
    - The goal is to turn the HTML content into a precise, readable, and informative summary that clearly represents the candidate’s profile.

    Return **only** the text of the summary, with no additional comments or formatting.

    Here is the HTML document to analyze:
    {text}
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
    graph = Neo4jGraph(refresh_schema=False)

    graph.query("""MATCH (n) DETACH DELETE n""")
    print("Graph cleared.")

    texts = []
    metadatas= []

    for filename in os.listdir(FILES_FOLDER):
        if filename.lower().endswith('.pdf'):
            filepath = os.path.join(FILES_FOLDER, filename)
            doc = fitz.open(filepath)
            print(f"Processo: {filename}")
            for page_number in range(len(doc)):
                page = doc[page_number]
                pix = page.get_pixmap()
                img_bytes = pix.tobytes(output='png')
                print(f"  Pagina {page_number+1}: {len(img_bytes)} bytes")

                image_base64 = base64.b64encode(img_bytes).decode('ascii')
                html = convert_img_to_html(image_base64)

                if "html" in html:
                    summary = summarize_document(html.replace("```html",'').replace("\n",'').replace("```",""))
                    texts.append(summary)
                    metadatas.append({"source": filename, "page": page_number + 1})

    embeddings_3_large : AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-3-large",
        openai_api_version="2024-12-01-preview",
        dimensions=3072
    )

    text_splitter = SemanticChunker(embeddings_3_large)
    documents = text_splitter.create_documents(texts,metadatas)

    for doc in documents:
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
    