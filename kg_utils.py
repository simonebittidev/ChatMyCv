import base64
from datetime import date
from typing import List
import fitz
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
import asyncio
from langchain_core.documents import Document
import os
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from models.llm_models import DocumentSummary, ChunkedSummary


FILES_FOLDER = "files/Simone Bitti"


def summarize_document(html_pages):
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4.1-mini",
        openai_api_version="2024-12-01-preview",
        temperature=0.0
    )

    text=str(html_pages)
   
    system_message = """
You are an expert at analyzing and summarizing HTML documents extracted from PDF pages, specifically CVs (resumes) and Cover Letters.

## Goal  
Your task is to read the full text of an HTML document—extracted from a PDF page that may represent a CV or a Cover Letter—and produce:
- A brief summary (2-3 sentences) that concisely describes the overall content and purpose of the document.
- An extremely detailed summary of the person described in the document. This summary must capture all relevant information, structure, and content from the source, with a strong focus on faithfully describing the individual, their background, experience, skills, and any other available details.

## Instructions

- Carefully analyze the entire HTML document, including all sections, headings, paragraphs, lists, and any other elements, recognizing the context of CVs and Cover Letters.
- For the brief summary, use only 2-3 sentences to capture the essence and intent of the document.
- For the detailed summary, be extremely thorough and focus on the person to whom the document refers. Do not omit important points, key examples, or supporting information about the individual, their achievements, career, education, skills, personality, or any other personal attribute or context found in the document.
- Use neutral, precise language. Do not rewrite, invent, or alter facts. Do not translate or interpret the content; just summarize faithfully.
- Focus on accuracy: if any part is ambiguous or unclear, include it in the summary as best as possible without making assumptions or fabrications.
- Do not omit any significant information or section.

## Output Format

Return your response in the following JSON format:
{
  "brief_overview": "<2-3 sentence summary of the document>",
  "detailed_summary": "<extremely detailed summary of all information about the person present in the document>"
}

The output must be only valid JSON, with no extra text, explanation, or commentary.
"""

    human_message = f"Here is the HTML document to summarize: {text}"
    
    summary = llm.with_structured_output(DocumentSummary).invoke([SystemMessage(content=system_message), HumanMessage(content=human_message)])
    
    return  summary

def convert_img_to_html(images):

    system_prompt = """You are an expert in analyzing images of PDF pages and converting their content into HTML format.

## Goal
You will be provided with a set of images, each representing a page from a PDF document. The images may contain text, images, tables, and other relevant information. Your task is to carefully analyze each image and extract all available information, generating a single, faithful HTML document that combines the content and structure from all pages.

## Instructions

- Carefully examine every element in every image, including text, headings, tables, images, and other graphical or structural components.
- Extract all visible and meaningful information from each image, and combine all information into a single HTML document that reflects the full content and structure of the original PDF.
- The combined HTML must preserve the structure and hierarchy as visually represented in the original PDF, maintaining the correct order and importance of information (e.g., headers, sections, lists, tables, images), as it appears page by page.
- Do not add colors, styles, custom classes, or any CSS. Only use plain HTML to represent the extracted information and structure.
- If any image is empty or does not contain any useful or readable information, ignore that image.
- The output should be only the complete, combined HTML document, with no extra explanation or commentary.

## Output Format

- Your response must contain only a single, valid HTML document that includes all the information and structure extracted from all the provided images, in the correct order.
- Do not include any explanations, comments, or content outside of the HTML.
"""

    
    messages = [SystemMessage(content=system_prompt)]

    for image in images:
        messages.append(HumanMessage(content=[
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}"
                    }
                    }
                ])
        )

    prompt = ChatPromptTemplate.from_messages(messages)
   
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4.1-mini",
        openai_api_version="2024-12-01-preview",
        temperature=0.0
    )

    chain = prompt | llm

    response = chain.invoke({})

    html = response.content
    
    return html

def get_summary_chunks(summary):
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4.1-mini",
        openai_api_version="2024-12-01-preview",
        temperature=0.0
    )
   
    chunking_prompt = """
You are an expert in document analysis and information structuring.

## Goal
Your task is to take a detailed summary of a CV or Cover Letter, and divide it into semantically meaningful chunks. Each chunk should represent a coherent topic, section, or set of related information. The objective is to ensure that each chunk is both self-contained and contextually meaningful, allowing for easy downstream processing and understanding.

## Instructions

- Analyze the provided summary and identify natural boundaries between sections or topics (e.g., "Education", "Work Experience", "Skills", "Certifications", "Personal Projects", etc.).
- Split the summary into chunks based on these boundaries. Avoid dividing information mid-sentence or mid-topic.
- Ensure that each chunk is long enough to be meaningful (preferably at least 3-4 sentences) but not excessively long (generally no more than 300 words per chunk).
- Retain all important context needed for each chunk to be understood independently, but do not repeat large blocks of text.
- If a section is particularly large, you may divide it into multiple logically connected chunks (e.g., "Work Experience (2018-2022)", "Work Experience (2023-present)").
- Maintain the original order of sections and information as they appeared in the summary.
- Do not omit any information or merge unrelated topics into a single chunk.
- For each chunk, identify exactly 5 keywords that best represent the content and main topic of the chunk (e.g., "CONTACT", "WORK EXPERIENCE", "EDUCATION", etc.).

## Output Format

Return your response in the following JSON format:

{
  "chunks": [
    {
      "title": "<short descriptive title for this chunk>",
      "content": "<text of this chunk>",
      "keywords": ["<KEYWORD1>", "<KEYWORD2>", "<KEYWORD3>"]
    },
    ...
  ]
}

- Each chunk must have a concise, descriptive title.
- The keywords field must be a list of exactly 3 uppercase keywords that summarize the content of the chunk.
- The output should be only valid JSON, with no extra text, explanation, or commentary.
"""



    human_message = f"Here the summary to analyze: {summary}"
    
    result = llm.with_structured_output(ChunkedSummary).invoke([SystemMessage(content=chunking_prompt), HumanMessage(content=human_message)])
    
    return  result.chunks

async def create_kg():
    graph = Neo4jGraph(refresh_schema=False)
    graph.query("""MATCH (n) DETACH DELETE n""")
    print("Graph cleared.")
    graph.close()

    metadatas= []

    for filename in os.listdir(FILES_FOLDER):
        if filename.lower().endswith('.pdf'):
            filepath = os.path.join(FILES_FOLDER, filename)
            doc = fitz.open(filepath)
            print(f"Processo: {filename}")

            images = []

            for page_number in range(len(doc)):
                page = doc[page_number]
                pix = page.get_pixmap()
                img_bytes = pix.tobytes(output='png')
                print(f"  Pagina {page_number+1}: {len(img_bytes)} bytes")

                image_base64 = base64.b64encode(img_bytes).decode('ascii')

                images.append(image_base64)
            
            html = convert_img_to_html(images)
            if html:
                summary = summarize_document(html.replace("```html",'').replace("\n",'').replace("```",""))

                embeddings_3_large : AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
                    azure_deployment="text-embedding-3-large",
                    openai_api_version="2024-12-01-preview",
                    dimensions=3072
                )

                chunks = get_summary_chunks(summary.detailed_summary) 
                documents = []
                for chunk in chunks:
                    metadatas = {
                        "embedding": embeddings_3_large.embed_query(chunk.content),
                        "source": filename,
                        "chunk_title": chunk.title,
                        "keywords": chunk.keywords
                    }
                    document = Document(page_content=f"{chunk.title} \n {chunk.content}", metadata = metadatas)
                    documents.append(document)

                llm = AzureChatOpenAI(
                    azure_deployment="gpt-4.1-mini",
                    openai_api_version="2024-12-01-preview",
                    temperature=0.0
                )

                today = date.today().strftime("%Y-%m-%d")

                brief_overview = f"""
                    All the HTML documents provided belong to the same original document (e.g., a multi-page CV or Cover Letter for a single person).
                    Do not treat them as separate entities. Instead, merge and analyze the content as a single document, preserving the correct order and overall structure.
                    Use the following brief overview as context to better understand the purpose and content:
                    Summary of the document: {summary.brief_overview}

                    Today date is: {today}
                    """

                llm_transformer = LLMGraphTransformer(
                    llm=llm,
                    allowed_nodes=["Profiency, Person, Role, Skill, ProgrammingLanguage, Technology, Organization, PersonalProject, Language, Concept, Contact, Certification, Activity, Project", "ProjectUrl", "DateRange"],
                    strict_mode=False,
                    additional_instructions=brief_overview
                )

                graph_documents = await llm_transformer.aconvert_to_graph_documents(documents)
                print(f"Nodes:{graph_documents[0].nodes}")
                print(f"Relationships:{graph_documents[0].relationships}")

                graph = Neo4jGraph(refresh_schema=False)
                graph.add_graph_documents(graph_documents, include_source=True, baseEntityLabel=False)      
                graph.close()

if __name__ == "__main__":
    asyncio.run(create_kg())
    