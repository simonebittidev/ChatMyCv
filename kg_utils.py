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

FILES_FOLDER = "files"

def summarize_document(html_pages):
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4.1",
        openai_api_version="2024-12-01-preview",
        temperature=0.0
    )

    text=str(html_pages)
   
    template=f"""
        Ti verrà fornito un documento in formato HTML che contiene informazioni. 
        
        Il tuo task è di analizzare attentamente la pagina HTML, estrapolare tutto il contenuto e elaborare un riassunto molto dettagliato del contenuto.

        IMPORTANTE: 
        - è essenziale che tutto il contenuto venga estrapolato dal documento HTML in modo che venga elaborato un riassunto dettagliato.

        Restituisci in output il testo del riassunto elaborato in base al contenuto del documento HTML.

        'Ecco il documento in formato HTML:{text}"""
    
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

    documents = []

    # for filename in os.listdir(FILES_FOLDER):
    #     if filename.lower().endswith('.pdf'):
    #         filepath = os.path.join(FILES_FOLDER, filename)
    #         doc = fitz.open(filepath)
    #         print(f"Processo: {filename}")
    #         for page_number in range(len(doc)):
    #             page = doc[page_number]
    #             pix = page.get_pixmap()
    #             img_bytes = pix.tobytes(output='png')
    #             print(f"  Pagina {page_number+1}: {len(img_bytes)} bytes")

    #             image_base64 = base64.b64encode(img_bytes).decode('ascii')
    #             html = convert_img_to_html(image_base64)

    #             if "html" in html:
    #                 summary = summarize_document(html.replace("```html",'').replace("\n",'').replace("```",""))
    #                 print(summary)
    #                 documents.append(Document(page_content=summary))


    summary = """Riassunto dettagliato del contenuto del documento HTML:

Il documento presenta il profilo professionale di Simone Bitti, un Software Developer.

**Dati personali e contatti:**
- Nome: Simone Bitti
- Professione: Software Developer
- Telefono: +39 3577480888
- Email: simone.bitti1996@gmail.com
- LinkedIn: https://www.linkedin.com/in/simone-bitti/
- GitHub: https://github.com/simonebittidev

**Profilo professionale:**
Simone è un Software Engineer fortemente orientato all’apprendimento continuo e al miglioramento personale. È abile nell’applicazione della metodologia Agile per la consegna di prodotti di alta qualità, appassionato di tutte le tecnologie e desideroso di rimanere aggiornato sulle ultime tendenze. Si impegna a collaborare con i team per raggiungere risultati ottimali e cerca di mettere a frutto le proprie competenze ed esperienze per guidare l’innovazione e contribuire alla crescita di organizzazioni dinamiche.

**Competenze (Skills):**
- Design Thinking
- Back End Coding
- Front End Coding
- Problem Solving
- Strong Communication

**Tecnologie conosciute:**
- .NET
- Azure
- CA
- Azure Storage
- Python
- LangChain
- LangGraph
- LangSmith
- Neo4j
- PromptFlow
- Postgres
- Redis
- Docker
- NOA
- SQL Server
- Cosmos DB
- GitHub
- SET
- LLMs
- Azure Cognitive Services
- Git
- Memcached
- Qdrant
- Azure Functions
- ITS
- REST API
- Bot Framework
- Azure DevOps
- Open Telemetry
- Application Insight

**Formazione (Education):**
- Luglio 2015: Diploma di maturità in Informatica presso l’Istituto Tecnico Odino Belluzzi.
- Settembre 2015: Ha frequentato i seguenti corsi Microsoft:
  - MOC 10264 - Developing Web Applications with Microsoft Visual Studio 2010
  - MOC 20462 - Administering Microsoft SQL Server Database
  - MOC 20466 - Implementing Data Models and Reports with Microsoft SQL Server

**Lingue (Languages):**
- Inglese: C1
- Inglese: B1

**Esperienze professionali (Experience):**

1. **Senior Software Engineer presso Vodafone (Novembre 2021 – in corso)**
   - Ha sviluppato soluzioni di chatbot AI per clienti (TOBi) e agenti di supporto (Agent Assistant).
   - TOBi: Assistente virtuale sviluppato con Microsoft Bot Framework e servizi Azure.
   - Super Agent: Strumento GenAI interno che utilizza LLM, pipeline RAG e knowledge graph.
   - Si è concentrato su architetture scalabili, integrazioni in tempo reale e deployment su cloud.

   - **TOBi (Novembre 2021 – Luglio 2024):**
     - Collaborazione nello sviluppo di chatbot con Microsoft Bot Framework e soluzioni cloud.
     - Creazione di pipeline Azure DevOps per il deployment continuo dei chatbot.
     - Sviluppo di Azure Functions per migliorare le funzionalità dei chatbot.
     - Utilizzo di Redis, Postgres, Memcached, Cosmos DB e servizi Azure per la gestione dei dati e il miglioramento delle performance.
     - Uso di Docker e Kubernetes per portabilità e scalabilità.
     - Integrazione di Azure Cognitive Services per arricchire le funzionalità dei chatbot e l’esperienza utente.
     - Partecipazione a code review e testing per mantenere standard qualitativi elevati.
     - Formazione dei membri del team su nuove tecnologie e processi.
     - Configurazione e manutenzione di sistemi IVR basati su Asterisk e integrazione con chatbot tramite API e script personalizzati.
     - Implementazione di protocolli di comunicazione socket-based per lo scambio dati in tempo reale tra client e server.
     - Sviluppo di applicazioni con tecnologie TTS (Text-to-Speech) e STT (Speech-to-Text) per interazioni vocali.
     - Contributi significativi a progetti open-source su GitHub tramite issue tracking, bug fixing e sviluppo di nuove funzionalità.

   - **Agent Assistant (Luglio 2024 – in corso):**
     - Costruzione di pipeline RAG avanzate usando Azure AI Search e Neo4j come vector store per il recupero strutturato.
     - Creazione di knowledge graph da dati non strutturati per migliorare la comprensione e la precisione del recupero.
     - Sviluppo di servizi Python modulari con LangChain e LangGraph per l’orchestrazione dinamica degli agenti.
     - Deployment e mantenimento di flussi tramite Microsoft PromptFlow, abilitando iterazione e monitoraggio continui.
     - Integrazione dei servizi Azure OpenAI (modelli GPT) con workflow aziendali sicuri per ragionamento basato su LLM.

2. **.Net Software Specialist - Consultant at Vodafone (Modis | Maggio 2020 - Novembre 2021)**
   - Applicazione dei principi di clean code per il refactoring del progetto, migrando da BotFramework 3 a BotFramework 4.
   - Sviluppo di unit test per aumentare la manutenibilità e leggibilità del codice.
   - Implementazione di vari design pattern (Decorator, Dependency Injection, Singleton, Domain) per migliorare architettura e scalabilità.
   - Collaborazione con il team di sviluppo per analisi dei requisiti e definizione delle specifiche tecniche.
   - Ottimizzazione delle performance tramite risoluzione bug, ottimizzazione del codice e testing approfondito.
   - Integrazione di sistemi esistenti con servizi di terze parti tramite API e web services.
   - Partecipazione attiva a meeting di progetto, fornendo suggerimenti per migliorare processi e delivery.

3. **Fullstack Developer (Progel | Settembre 2015 - Ottobre 2019)**
   - Responsabile dello sviluppo e manutenzione di applicazioni web in ambito sanitario usando .Net Framework, C#, Vue.Net.
   - Sviluppo e manutenzione di applicazioni in .Net Framework e .Net Core con Asp.Net.
   - Gestione e sviluppo di storage dati con Microsoft SQL Server (creazione tabelle, viste, stored procedure per analisi dati).
   - Manutenzione di soluzioni front-end con JavaScript, TypeScript, HTML, CSS.
   - Utilizzo di Azure DevOps per versionamento e branching del codice.
   - Utilizzo di pratiche Scrum per la gestione dei task (Scrum board).

**Conclusione:**
Simone Bitti è un software engineer con una solida esperienza nello sviluppo di chatbot AI, soluzioni cloud, architetture scalabili e tecnologie moderne sia back-end che front-end. Ha lavorato in contesti enterprise, contribuendo a progetti innovativi e open-source, con una forte propensione all’apprendimento continuo, collaborazione e miglioramento dei processi di sviluppo."""
    
    # documents.append(Document(page_content=summary))


    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    # documents = text_splitter.split_documents(documents)

    # embeddings_3_large : AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    #     azure_deployment="text-embedding-3-large",
    #     openai_api_version="2024-12-01-preview",
    #     dimensions=3072
    # )
    # text_splitter = SemanticChunker(embeddings_3_large)
    # documents = text_splitter.create_documents([summary])
    # print(documents[0].page_content)
  
    graph = Neo4jGraph(refresh_schema=False)

    # llm = AzureChatOpenAI(
    #     azure_deployment="gpt-4.1",
    #     openai_api_version="2024-12-01-preview",
    #     temperature=0.0
    # )

    # llm_transformer = LLMGraphTransformer(llm=llm)

    # graph_documents = await llm_transformer.aconvert_to_graph_documents(documents)
    # print(f"Nodes:{graph_documents[0].nodes}")
    # print(f"Relationships:{graph_documents[0].relationships}")

    # graph.add_graph_documents(graph_documents, include_source=True, baseEntityLabel=False)      
    # graph.close()

    # graph.refresh_schema()
    # print(graph.schema)

    # enhanced_graph = Neo4jGraph(enhanced_schema=True)

    llm = AzureChatOpenAI(
        azure_deployment="gpt-4.1",
        openai_api_version="2024-12-01-preview",
        temperature=0.0
    )

    # chain = GraphCypherQAChain.from_llm(
    #     graph=enhanced_graph, llm=llm, verbose=True, allow_dangerous_requests=True
    # )
    
    # response = chain.invoke({"query": "Quali lingue parla simone?"})

    # response = chain.invoke({"query": "In quali aziende ha lavorato Simone?"})

    # response = chain.invoke({"query": "Quando ha lavorato in progel?"})

    # response = chain.invoke({"query": "Cosa fa in Vodafone e quali tecnologie usa?"})
    
    # print(response)

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

    # Extract entities from text
    class Entities(BaseModel):
        """Identifying information about entities."""

        names: List[str] = Field(
            ...,
            description="All the person, organization, or business entities  that " "appear in the text",
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

    # entity_chain.invoke({"question": "Qual è la email di simone?"}).names

    graph.query(
    "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

    def generate_full_text_query(input: str) -> str:
        """
        Generate a full-text search query for a given input string.

        This function constructs a query string suitable for a full-text
        search. It processes the input string by splitting it into words and 
        appending a similarity threshold (~2 changed characters) to each
        word, then combines them using the AND operator. Useful for mapping
        entities from user questions to database values, and allows for some 
        misspelings.
        """
        full_text_query = ""
        words = [el for el in input.split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()
    
    def structured_retriever(question: str) -> str:
        """
        Collects the neighborhood of entities mentioned
        in the question
        """
        result = ""
        entities = entity_chain.invoke({"question": question})
        for entity in entities.names:
            response = graph.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, 
                {limit:2})
                YIELD node,score
                CALL {
                MATCH (node)-[r:!MENTIONS]->(neighbor)
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS 
                output
                UNION
                MATCH (node)<-[r:!MENTIONS]-(neighbor)
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS 
                output
                }
                RETURN output LIMIT 50
                """,
                {"query": generate_full_text_query(entity)},
            )
            result += "\n".join([el['output'] for el in response])
        return result
    
    def retriever(question: str):
        print(f"Search query: {question}")
        structured_data = structured_retriever(question)
        unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
        final_data = f"""Structured data:
        {structured_data}
        Unstructured data:
        {"#Document ". join(unstructured_data)}
            """
        return final_data
        
    template = """
    Sei un assistente virtuale che si occupa di rispondere a domande riguardo informazioni lavorative o generiche di una persona.

    Ti verranno fornite le seguenti informazioni:
    - l'input dell'utente
    - un contesto composto da Structured Data derivata da un knwoledge graph e Unstructured Data ricavato da documenti presenti su un vectore store.
        
    Il tuo task è di analizzare il contesto fornito ed elaborare una risposta finale per l'utente.

    Segui le seguenti istruzioni:
    - Basati **esclusivamente** sulle informazioni di contesto per elaborare la risposta finale per l'utente.
    - Rispondi direttamente all'utente senza menzionare da dove hai ricavato le informazioni.
    - Usa un amichevole.

    
    Question: {question}

    Contesto: {context}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        RunnableParallel(
            {
                "context": RunnableLambda(lambda x: x["question"]) | retriever,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # result = chain.invoke({"question": "Dove ha lavorato Simone?"})
    # result = chain.invoke({"question": "Esattamente in progel di cosa si è occupato?"})
    # result = chain.invoke({"question": "Esattamente in Vodafone di cosa si è occupato?"})
    result = chain.invoke({"question": "Dimmi tutto quello che sai"})

    print(result)

if __name__ == "__main__":
    asyncio.run(create_kg())
    