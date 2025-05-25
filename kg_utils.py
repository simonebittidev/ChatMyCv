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
        Ti verrà fornito un documento in formato HTML che contiene informazioni rigurdanti un curriculum vitae. 
        
        Il tuo task è di analizzare attentamente il documento fornito, estrapolare tutto il contenuto e elaborare un riassunto molto dettagliato del contenuto.

        Segui le seguiti istruzioni: 
        - è **essenziale** che tutto il contenuto venga estrapolato dal documento HTML.
        - crea un riassunto il più dettagliato possibile senza tralasciare niente.
        - correggi o arricchisci il riassunto cercando di rendere più chiare le esperienze e le tecnologie a cui fa riferimento.
        - non aggiungere al riassunto esperienze o tecnologie che non sono presenti nel documento originale.

        Restituisci in output solamente il testo contente riassunto elaborato a partire dal documento fornito in input.

        Ecco il documento HTML da analizzare: {text}"""
    
    ai_msg = llm.invoke([HumanMessage(content=template)])
    response  = ai_msg.content
    
    try:
        return  response
    except json.JSONDecodeError as e:
        print("Failed to decode JSON:", e)
        return None


def convert_img_to_html(encoded_image):

    system_prompt = """Ti verrà fornita una immagine che rappresenta una pagina di un PDF e che può contenenere testo, immagini, tabelle e altre informazioni.
    Il tuo compito è di analizzare attentamente l'immagine ed estrapolare tutte le informazioni in formato HTML.

    Segui le seguenti istruzioni:
    - Se l'immagine è vuota o non contiene nessuna informazioni allora produrre nulla in output.
    - Non aggiungere colori o stili nell'HTML generato.
    - L'HTML generato deve manterele la stessa struttura rappresentata nell'immagine in modo da mantenere l'importanza e l'ordine delle varie informazioni estrapolate.

    Restituisci in output solamente l'HTML generato in base all'immagine data in input.
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
        doc.page_content = f"This document is a chunk of the original document {doc.metadata['source']} and refers to Simone Bitti. \n {doc.page_content}."
  
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
    