# AskMyCv

A personal study project that turns your CV and cover letter into an interactive knowledge graph you can query via a chat interface.

---

## 🚀 Overview

**AskMyCv** is a chat-bot application that lets you explore my professional background, skills and projects by asking natural-language questions. Under the hood:

1. **Extraction** of entities and relationships from PDF CVs, cover letters—and soon GitHub READMEs—using a Large Language Model.  
2. **Storage** in a Neo4j knowledge graph, where experiences, skills and projects become connected nodes.  
3. **Querying**: incoming user questions are translated into Cypher queries by an LLM, then executed against the graph.  
4. **Real-time streaming** of answers via Server-Sent Events (SSE).  
5. **Orchestration** of data flows and chat logic with LangGraph.

This project is purely experimental, a playground for combining NLP, graph databases and modern backend/frontend tooling.

---

## 🔧 Tech Stack

- **Backend**:  
  - Python 3.x, [FastAPI](https://fastapi.tiangolo.com/)  
  - Neo4j Aura (cloud-hosted)  
  - APScheduler (to keep the Aura instance “warm”)  
  - LangGraph (flow orchestration)  
  - OpenAI GPT (or any LLM) for NLP tasks  

- **Frontend**:  
  - Next.js / React / TypeScript  
  - Server-Sent Events for live response streaming  
  - Tailwind CSS & Heroicons  

---

## 📦 Installation

### 1. Clone the repo
```bash
git clone https://github.com/your-username/AskMyCv.git
cd AskMyCv
```

### 2. Backend setup
  1.	Create a virtual environment and install dependencies:

  ```bash
  python -m venv .venv
  source .venv/bin/activate      # macOS/Linux
  .venv\Scripts\activate         # Windows
  pip install -r requirements.txt
  ```

  2.	Set environment variables in a .env file:

  ```dotenv
  NEO4J_URI=<your-neo4j-aura-uri>
  NEO4J_USER=<username>
  NEO4J_PASSWORD=<password>
  OPENAI_API_KEY=<your-openai-key>
  ```

  3.	Start the FastAPI server:

  ```bash
  uvicorn app:app --reload
  ```

### 3. Frontend setup

```bash
cd client
npm install
npm run dev
```

## ⚙️ Usage
	1.	Open your browser at http://localhost:3000.
	2.	You’ll see suggested questions on first load (e.g. “What technologies does Simone have experience with?”).
	3.	Type or click a suggestion—responses will stream back in real time.
	4.	Ask anything about my CV, cover letter or projects!

## 🛠 Pipeline Details
	1.	Document Parsing
	    •	PDFs (CV & cover letter) are parsed into text.
	    •	Soon: GitHub README files will be ingested directly via API.
	2.	Entity & Relation Extraction
	    •	LLM prompts identify experiences, skills, technologies and project names.
	    •	Relationships (e.g. “worked at → Company X”, “used → Technology Y”) become graph edges.
	3.	Graph Construction
	    •	Neo4j nodes represent entities; edges represent relations.
	    •	Custom schema defined via neo4j_graph = Neo4jGraph(enhanced_schema=True).
	4.	Query Translation
	    •	User questions → LLM prompt → generated Cypher query.
	    •	Queries executed via Neo4j driver; results streamed back.
	5.	Real-time Chat
	    •	SSE endpoint (/stream) sends incremental message chunks.
	    •	Frontend renders them as they arrive for a fluid UX.

⸻

## 🤝 Contributing

This is a solo study project, but I’d love to hear your ideas!
	•	Report issues or suggest features via GitHub Issues.
	•	Feel free to submit pull requests if you improve extraction prompts, graph models or frontend UX.

⸻

## 📜 License

Distributed under the MIT License. See LICENSE for more details.


 
