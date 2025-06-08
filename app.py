import json
from typing import List
import os
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from utils import get_data
import ssl
from langchain.callbacks.base import BaseCallbackHandler
from datetime import date
from langchain_core.messages import HumanMessage, SystemMessage

class RewrittenQuestion(BaseModel):
    rewritten_question: str = Field(
        description="The clarified and rephrased version of the user's question, rewritten for maximum clarity based on conversation context."
    )

print(ssl.OPENSSL_VERSION)

load_dotenv()

app = FastAPI()

print(f"NEO4J_URI: {os.getenv('NEO4J_URI')}")
print(f"NEO4J_PASSWORD: {os.getenv('NEO4J_PASSWORD')}")
print(f"NEO4J_USERNAME: {os.getenv('NEO4J_USERNAME')}")

# Serve React app (build)
client_path = Path("client/out")
app.mount("/static", StaticFiles(directory=client_path), name="static")

class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.queue = []

    def on_llm_new_token(self, token: str, **kwargs):
        self.queue.append(token)

    def get(self):
        while self.queue:
            yield self.queue.pop(0)

class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities  that " "appear in the text",
    )

@app.get("/stream")
async def stream_sse(text: str, history: str):
    print(f"Received text: {text}")
    print(f"Received history: {history}")

    message=text
    history = json.loads(history)

    llm_history = AzureChatOpenAI(
        azure_deployment="gpt-4.1",
        openai_api_version="2024-12-01-preview",
        temperature=0.7 #more creative and less deterministic responses
    )

    rewrite_question_prompt = """
You are an expert assistant in communication clarity and context analysis.

## Goal
Your task is to analyze the user's latest question together with the conversation history. If the question is unclear or ambiguous, you must use the conversation history to rewrite it in a way that makes it clearer and more precise, while preserving the original intent and all important details. If the question is already clear, simply return it as is.

## Instructions

- Carefully read the user's question and the conversation history.
- If the question is ambiguous, vague, or hard to understand, rewrite it to be as clear and precise as possible, using relevant context from the conversation history to clarify meaning.
- Do not invent or add information that is not present in the conversation.
- If the question is already clear and unambiguous, return it exactly as it was given.
- Always use the same language as the user's question.
- Do not answer the question or provide any additional information—your only task is to rewrite it if needed.

## Output Format

Return your response in the following JSON format:

{
  "rewritten_question": "<the clarified version of the user's question, or the original if it was already clear>"
}

- The output must be only valid JSON, with no extra explanation or commentary.
"""

    messages = [SystemMessage(content=rewrite_question_prompt), HumanMessage(content=f"User's input: {text}\nChat history: {history}")]

    result = llm_history.with_structured_output(RewrittenQuestion).invoke(messages)
    message = result.rewritten_question
    print(f"Received message: {message}")
    print(f"uri {os.getenv('NEO4J_URI')}")

    enhanced_graph = Neo4jGraph(enhanced_schema=True)

    handler = StreamHandler()

    llm = AzureChatOpenAI(
        azure_deployment="gpt-4.1",
        openai_api_version="2024-12-01-preview",
        temperature=0.7 #more creative and less deterministic responses
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

    text2cypher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Given an input question, convert it to a Cypher query. No pre-amble. "
                "Do not wrap the response in any backticks or anything else. Respond with Cypher statements only!"
            ),
        ),
        (
            "human",
            (
                """You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.
Do not wrap the response in any backticks or anything else. Respond with Cypher statements only!

Here is the schema information
{schema}

User input: {question}

Return your answer in the following JSON format:
{{
  "main_query": "<a Cypher query to answer the user's question>",
  "document_distinct_query": "<a Cypher query that, using the same pattern as main_query, returns an array of strings, each string being the full text (for example, the 'text' property) of a DISTINCT Document node from which the main_query results are extracted. The output of this query must be a list of strings, each containing the full text of a relevant document.>"
}}
"""
            ),
        ),
    ]
)



    text2cypher_chain = text2cypher_prompt | llm | StrOutputParser()

    today = date.today().strftime("%Y-%m-%d")

    context = get_data(enhanced_graph, vector_index, message, text2cypher_chain)

    template = f"""You are a virtual assistant of Ask my cv, a tool designed to answer questions from the user about Simone.

You will be provided with the following information:

- The user’s question.
- A chat history containing previous interactions with the user useful to understand better the context of the question.
- A context consisting of:
    1. Structured Data derived from a knowledge graph.
    2. Unstructured Data extracted from documents stored in a vector database, all related to the candidate in question.

Your goal is to analyze the provided context and craft a final response that highlights the strengths of the candidate.

How to respond:
- Use **only** the information provided in the context.
- Do **not** mention sources or explain how you derived the information. Just answer the question as if you already know the candidate.
- Structure your response clearly:
    - Start with a concise summary or direct answer.
    - Use paragraphs to separate different ideas.
    - Highlight important skills, achievements, or traits in **bold**.
    - Use **Markdown formatting only** for clarity and readability.
    - Use bullet points (`-`) for lists.
- Rephrase and summarize the information — **do not copy-paste**, and **do not invent**.
- Never exaggerate Simone’s role or achievements. Only mention responsibilities, leadership, or specific results if they are clearly stated in the context provided. Do not attribute to Simone roles such as “project leader”, “manager”, or “having guided a team” unless this is explicitly mentioned in the context. Always keep the answer grounded and aligned with the information available. If uncertain or if the information is not in the context, specify clearly that this detail is not present or indicated in the provided context, rather than refusing to answer.
- Your tone should be natural and friendly, as if helping the user quickly understand Simone's profile and experiences.
- The final response must be clear, polished, and make Simone's strengths and experiences easy to understand and appreciate.
- Never mention the context or never explain the type of answer you are providing (e.g., "Risposta ironica e difensiva:", "Risposta ironica e difensiva:"). Just provide the answer directly.

**Special instruction for chitchat or non-job-related questions:**
- If the user asks a casual, playful, or off-topic question (chitchat), respond with an ironic, witty tone, always defending or supporting Simone. Support Simone, but avoid exaggerations or over-the-top praise. Be relatable and use a touch of humor, but stay credible.
- If the user asks who you are, respond indicating that you are a virtual assistant of Ask my cv, a tool designed to answer questions from the user about Simone.

**Special instruction for technology-related questions:**  
- If the user asks about the technologies used by Simone (such as programming languages, frameworks, tools, or platforms), provide a well-structured and organized answer. Whenever possible, divide the technologies into clear categories such as:
    - Backend
    - Frontend
    - Database
    - DevOps / Cloud
    - Tools and other relevant areas
- Use bullet points for each category, and make the list easy to read and understand, highlighting Simone’s experience with each technology where available.

User Input: {message}

Context: {context}

It is **essential** to answer using the same language as the user, so if the user asks a question in Italian, you must answer in Italian, and if the user asks a question in English, you must answer in English.

Today date is: {today}
        """

    llm_stream = AzureChatOpenAI(
        azure_deployment="gpt-4.1",
        openai_api_version="2024-12-01-preview",
        streaming=True,
        callbacks=[handler],
        temperature=0.7 #more creative and less deterministic responses
    )

    async def event_generator():
        llm_stream.invoke([SystemMessage(content=template)])
        for token in handler.get():
            yield f"data: {json.dumps({'role': 'ai', 'content': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

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
