# =====================================================
# IMPORTS
# =====================================================
import streamlit as st
import pandas as pd
import io, json, re, zipfile, ast, time, uuid, sqlite3
from typing import List, Dict, Any
from contextlib import redirect_stdout
from pathlib import Path
import google.generativeai as genai
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import chromadb
from sentence_transformers import CrossEncoder
import time
import logging
from enum import Enum
from collections import defaultdict
from pydantic import BaseModel
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("RAG")

# =====================================================
# Retrieval Enum + Pydantic
# =====================================================

class RetrievalQuality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RetrievalMetrics(BaseModel):
    score: float
    quality: RetrievalQuality
    top_score: float
    mean_score: float

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# DATABASE (TRACEABILITY)
# =====================================================
DB_PATH = "agentic_rag.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
 
    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        trace_id TEXT,
        session_id TEXT,
        user_query TEXT,
        route TEXT,
        final_answer TEXT,
        retrieval_score REAL,
        retrieval_quality TEXT,
        answer_score REAL,
        latency REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Migration: Add session_id if it doesn't exist (for existing databases)
    try:
        cur.execute("ALTER TABLE conversations ADD COLUMN session_id TEXT")
    except sqlite3.OperationalError:
        pass # Column already exists
 
    cur.execute("""
    CREATE TABLE IF NOT EXISTS tool_calls (
        trace_id TEXT,
        tool_name TEXT,
        input TEXT,
        output TEXT,
        latency REAL,
        success INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
 
    cur.execute("""
    CREATE TABLE IF NOT EXISTS llm_calls (
        trace_id TEXT,
        model TEXT,
        purpose TEXT,
        prompt TEXT,
        response TEXT,
        latency REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
 
    cur.execute("""
    CREATE TABLE IF NOT EXISTS semantic_cache (
        query_text TEXT,
        query_embedding BLOB,
        response_json TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
 
    conn.commit()
    conn.close()

def log_db(table, columns, values):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    placeholders = ",".join(["?"] * len(values))
    cols = ",".join(columns)
    cur.execute(
        f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
        values
    )
    conn.commit()
    conn.close()

init_db()

# =====================================================
# UTILITIES
# =====================================================
def extract_json(text: str):
    try:
        return json.loads(re.search(r"(\{.*\})", text, re.DOTALL).group(1))
    except:
        return None

# def score_retrieval(scores: List[float]):
#     if not scores:
#         return 0.0, "None"
    
#     max_score = max(scores)
#     avg_top3 = sum(sorted(scores, reverse=True)[:3]) / min(len(scores), 3)
    
#     if max_score > 0.8 or avg_top3 > 0.6:
#         return round(max_score, 2), "High"
#     if max_score > 0.5 or avg_top3 > 0.3:
#         return round(max_score, 2), "Medium"
#     return round(max_score, 2), "Low"

def score_retrieval(scores):

    scores = np.array(scores)

    # Softmax normalize logits
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / exp_scores.sum()

    confidence = float(probs.max())
    top = float(scores.max())

    if confidence < 0.45:
        quality = RetrievalQuality.LOW
    elif confidence < 0.70:
        quality = RetrievalQuality.MEDIUM
    else:
        quality = RetrievalQuality.HIGH

    return RetrievalMetrics(
        score=confidence,
        quality=quality,
        top_score=top,
        mean_score=float(scores.mean())
    )

# =====================================================
# SAFE PANDAS EXECUTION
# =====================================================
def execute_pandas_code(code: str, df: pd.DataFrame):
    safe_globals = {
        "df": df,
        "pd": pd,
        "json": json,
        "ast": ast,
        "__builtins__": {
            "print": print, "len": len, "range": range,
            "list": list, "dict": dict, "set": set,
            "sum": sum, "min": min, "max": max,
            "round": round, "abs": abs,
            "__import__": lambda n,*a: __import__(n)
            if n in ["pandas","numpy","math","datetime","ast","re"]
            else (_ for _ in ()).throw(ImportError("Blocked"))
        }
    }
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            exec(code, safe_globals, {})
        out = buf.getvalue().strip()
        if out:
            return out
        result = eval(code, safe_globals, {})
        if isinstance(result, pd.DataFrame):
            return result.to_markdown(index=False)
        if isinstance(result, pd.Series):
            return json.dumps(result.dropna().tolist()[:100])
        if isinstance(result, (list, dict, tuple, set)):
            return json.dumps(list(result), indent=2)
        return str(result)
    except Exception as e:
        return f"EXECUTION ERROR: {e}"

# =====================================================
# DATA INTROSPECTOR
# =====================================================
class DataIntrospector:
    def __init__(self, df):
        self.df = df

    def context(self):
        ctx = f"Rows={len(self.df)}, Columns={len(self.df.columns)}\n"
        for c in self.df.columns:
            ctx += f"- {c} ({self.df[c].dtype})\n"
        ctx += "\nSample:\n" + self.df.head(3).to_markdown(index=False)
        return ctx

# =====================================================
# CORE ENGINE
# =====================================================
class UnifiedDataEngine:
    def __init__(self, api_key):
        # self.client = genai.Client(api_key=api_key)
        genai.configure(api_key=api_key)
        self.llm = genai.GenerativeModel("gemini-2.5-flash")
        self.df = None
        self.all_chunks = []

        self.chroma = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma.get_or_create_collection("unified_storage")

        self.splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        self.bm25 = None
        self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.history = []
        self.cache_threshold = 0.95  # High threshold for semantic similarity

    # ================= INGESTION =================
    def ingest_files(self, files):
        pdfs = []
        for f in files:
            if f.name.endswith(".csv"):
                self.df = pd.read_csv(f)
            elif f.name.endswith(".xlsx"):
                self.df = pd.read_excel(f)
            elif f.name.endswith(".pdf"):
                pdfs.append(f)
        if pdfs:
            self._ingest_pdfs(pdfs)

    def _ingest_pdfs(self, pdfs):
        for pdf in pdfs:
            reader = PdfReader(pdf)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text:
                    continue
                chunks = self.splitter.split_text(text)
                embeds = genai.embed_content(model="gemini-embedding-001", content=chunks)
                
                self.collection.add(
                    ids=[f"{pdf.name}_{i}_{j}" for j in range(len(chunks))],
                    embeddings=embeds["embedding"],
                    documents=chunks,
                    metadatas=[{"source": pdf.name, "page": i}] * len(chunks)
                )
                self.all_chunks.extend(chunks)
        self.bm25 = BM25Okapi([c.split() for c in self.all_chunks])

    # ================= PANDAS AGENT =================
    def pandas_agent(self, query: str, trace_id: str, history: List[Dict[str, str]] = None):
        """
        Intelligent Pandas agent with:
        - ambiguity handling
        - safe pandas execution
        - text analysis routing
        - explanation summarization
        - full traceability
        """

        introspector = DataIntrospector(self.df)
        history_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in (history or [])])

        # ---------------------------
        # 1. Build system prompt
        # ---------------------------
        system_prompt = f"""
            You are an expert data analyst working with a Pandas DataFrame called df.

            RULES:
            - Use ONLY pandas code
            - Never hallucinate column names
            - If query is ambiguous, ask a clarifying question
            - For counts, use len(df) unless user specifies an ID
            - Wrap all boolean conditions in parentheses
            - Return small tables only (head / aggregates)
            - For text analysis, return a list/Series of strings

            DATA CONTEXT:
            {introspector.context()}

            CONVERSATION HISTORY:
            {history_str}

            RESPONSE FORMAT (JSON ONLY):

            {{ "clarification": "question" }}

            OR

            {{
            "tool": "execute_pandas_code",
            "arguments": {{
                "code": "python pandas code here"
            }}
            }}

            OR

            {{ "final_answer": "text explanation" }}

            USER QUERY:
            {query}
            """

        # ---------------------------
        # 2. LLM reasoning call
        # ---------------------------
        t0 = time.time()
        # res = self.client.models.generate_content(
        #     model="gemini-2.5-flash",
        #     contents=system_prompt
        # )
        res = self.llm.generate_content(system_prompt)
        llm_latency = time.time() - t0

        # Log LLM reasoning
        log_db(
            "llm_calls",
            ["trace_id", "model", "purpose", "prompt", "response", "latency"],
            (trace_id, "gemini", "pandas", system_prompt, res.text, llm_latency)
        )

        parsed = extract_json(res.text)
        if not parsed:
            return "I could not understand the request clearly. Could you rephrase?"

        # ---------------------------
        # 3. Clarification path
        # ---------------------------
        if "clarification" in parsed:
            return parsed["clarification"]

        # ---------------------------
        # 4. Tool execution path
        # ---------------------------
        if parsed.get("tool") == "execute_pandas_code":
            code = parsed["arguments"]["code"]

            print(f"\n[PANDAS CODE][TRACE_ID={trace_id}]\n{code}\n")

            t1 = time.time()
            result = execute_pandas_code(code, self.df)
            tool_latency = time.time() - t1

            # Log tool execution
            log_db(
                "tool_calls",
                ["trace_id", "tool_name", "input", "output", "latency", "success"],
                (trace_id, "pandas", code, result, tool_latency, 1)
            )

            # ---------------------------
            # 4a. Text-analysis follow-up
            # ---------------------------
            try:
                parsed_result = json.loads(result)
                if (
                    isinstance(parsed_result, list)
                    and parsed_result
                    and all(isinstance(x, str) for x in parsed_result)
                ):
                    analysis_prompt = f"""
                    Summarize the key themes across the following support interaction summaries.
                    IMPORTANT:
                    - Do NOT count individual word occurrences.
                    - Treat each summary as one interaction.
                    - Do NOT report numeric frequencies unless explicitly asked.
                    - Focus on qualitative patterns and recurring issue categories.


                    QUESTION:
                    {query}

                    TEXT:
                    {"---".join(parsed_result[:50])}
                    """
                    t2 = time.time()
                    analysis_res = self.llm.generate_content(analysis_prompt)
                    analysis_latency = time.time() - t2

                    # Log text-analysis LLM call
                    log_db(
                        "llm_calls",
                        ["trace_id", "model", "purpose", "prompt", "response", "latency"],
                        (
                            trace_id,
                            "gemini",
                            "pandas_text_analysis",
                            analysis_prompt,
                            analysis_res.text,
                            analysis_latency
                        )
                    )

                    return analysis_res.text
            except Exception:
                pass

            # ---------------------------
            # 4b. Explanation / summarization
            # ---------------------------
            explain_prompt = f"""
                Explain the following output clearly and concisely.

                OUTPUT:
                {result}
                """
            t3 = time.time()
            explain_res = self.llm.generate_content(explain_prompt)

            explain_latency = time.time() - t3

            # Log explanation call
            log_db(
                "llm_calls",
                ["trace_id", "model", "purpose", "prompt", "response", "latency"],
                (
                    trace_id,
                    "gemini",
                    "pandas_explanation",
                    explain_prompt,
                    explain_res.text,
                    explain_latency
                )
            )

            return explain_res.text

        # ---------------------------
        # 5. Direct final answer path
        # ---------------------------
        return parsed.get("final_answer", "I could not generate a valid response.")

    # ================= RAG AGENT =================
    # def rag_agent(self, query, trace_id, history: List[Dict[str, str]] = None):
    #     history_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in (history or [])])
        
    #     # Rewrite query for better retrieval if history exists
    #     if history:
    #         rewrite_prompt = f"Given the following conversation history and a new user query, rewrite the query to be a standalone question that can be used for retrieval. If it's already standalone, return it as is.\n\nHistory:\n{history_str}\n\nNew Query: {query}\n\nStandalone Query:"
    #         rewrite_res = self.llm.generate_content(rewrite_prompt)
    #         search_query = rewrite_res.text.strip()
    #     else:
    #         search_query = query

    #     q_emb = genai.embed_content(model="models/text-embedding-004", content=search_query)["embedding"]

    #     vec = self.collection.query(query_embeddings=[q_emb], n_results=10)
    #     bm25_docs = self.bm25.get_top_n(search_query.split(), self.all_chunks, n=5)
    #     candidates = list(set(vec["documents"][0] + bm25_docs))

    #     scores = self._reranker.predict([[search_query, c] for c in candidates])
    #     ranked = sorted(zip(scores, candidates), reverse=True)[:5]

    #     retrieval_score, retrieval_quality = score_retrieval(scores.tolist())

    #     context = "\n---\n".join(c for _,c in ranked)

    #     prompt = f"Context:\n{context}\n\nConversation History:\n{history_str}\n\nQuestion:{query}"
    #     t0 = time.time()
    #     res = self.llm.generate_content(prompt)


    #     log_db("llm_calls", ["trace_id","model","purpose","prompt","response","latency"], (trace_id,"gemini","rag",prompt,res.text,time.time()-t0))

    #     return res.text, retrieval_score, retrieval_quality

    def rag_agent(self, query, trace_id, history=None):

        history = history or []
        history_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history])

        logger.info(f"[User Query] {query}")

        # ---------------------------------------------------
        # 1. Rewrite query
        # ---------------------------------------------------
        rewrite_prompt = f"""
    Rewrite the user query for semantic retrieval.
    Preserve entities and technical terms.
    Remove chit-chat.

    Conversation:
    {history_str}

    User Query:
    {query}

    Standalone retrieval query:
    """

        rewrite_res = self.llm.generate_content(rewrite_prompt)
        base_query = rewrite_res.text.strip()

        logger.info(f"[Rewrite] Base query: {base_query}")

        # ---------------------------------------------------
        # 2. Multi-query expansion
        # ---------------------------------------------------
        expand_prompt = f"""
    Generate 4 alternative search queries capturing different
    phrasings and aspects of:

    {base_query}

    Return newline list.
    """

        expand_res = self.llm.generate_content(expand_prompt)

        expanded_queries = [
            q.strip("- ").strip()
            for q in expand_res.text.split("\n")
            if q.strip()
        ]

        search_queries = [base_query] + expanded_queries
        logger.info(f"[Expansion] Queries: {search_queries}")

        # ---------------------------------------------------
        # 3. Hybrid retrieval
        # ---------------------------------------------------
        candidate_scores = defaultdict(float)

        for sq in search_queries:

            q_emb = genai.embed_content(
                model="gemini-embedding-001",
                content=sq
            )["embedding"]

            vec = self.collection.query(query_embeddings=[q_emb], n_results=8)

            for rank, doc in enumerate(vec["documents"][0]):
                candidate_scores[doc] += 1.0 / (rank + 1)

            bm25_docs = self.bm25.get_top_n(sq.split(), self.all_chunks, n=5)

            for rank, doc in enumerate(bm25_docs):
                candidate_scores[doc] += 0.8 / (rank + 1)

        candidates = list(candidate_scores.keys())
        logger.info(f"[Retrieval] Candidates: {len(candidates)}")

        # ---------------------------------------------------
        # 4. Cross-encoder rerank
        # ---------------------------------------------------
        rerank_pairs = [[base_query, c] for c in candidates]
        ce_scores = self._reranker.predict(rerank_pairs)

        reranked = sorted(
            zip(ce_scores, candidates),
            reverse=True
        )[:8]

        for i, (score, chunk) in enumerate(reranked[:5]):
            logger.info(f"[Rerank] #{i+1} score={score:.4f} | {chunk[:120]}...")

        # ---------------------------------------------------
        # 5. Confidence scoring
        # ---------------------------------------------------
        metrics = score_retrieval(ce_scores.tolist())
        logger.warning(f"[Confidence] {metrics.model_dump()}")

        if metrics.quality is RetrievalQuality.LOW:
            logger.error("[Guardrail] Retrieval LOW → refusing answer")
            return (
                "I don't have enough reliable information to answer this confidently.",
                metrics.score,
                metrics.quality.value
            )

        # ---------------------------------------------------
        # 6. Context compression
        # ---------------------------------------------------
        unique_chunks = []
        seen = set()

        for score, chunk in reranked:
            key = hash(chunk[:200])
            if key not in seen:
                seen.add(key)
                unique_chunks.append((score, chunk))

        context_blocks = []

        for i, (score, chunk) in enumerate(unique_chunks[:5], 1):
            context_blocks.append(
                f"[Source {i} | relevance {round(score, 3)}]\n{chunk}"
            )

        context = "\n\n".join(context_blocks)

        logger.info("[Context Preview]")
        logger.info(context[:800])

        # ---------------------------------------------------
        # 7. Grounded generation
        # ---------------------------------------------------
        answer_prompt = f"""
    You must answer ONLY using the provided context.
    If insufficient, say you don't know.

    Context:
    {context}

    Conversation:
    {history_str}

    Question:
    {query}

    Grounded answer:
    """

        logger.info("[Generation] Sending to LLM")

        t0 = time.time()

        res = self.llm.generate_content(answer_prompt)

        latency = time.time() - t0

        logger.info("[Generation] Response received")

        log_db(
            "llm_calls",
            ["trace_id","model","purpose","prompt","response","latency"],
            (trace_id, "gemini", "rag", answer_prompt, res.text, latency)
        )

        return res.text, metrics.score, metrics.quality.value


    # ================= ANSWER CRITIC =================

    def answer_critic(self, query, answer, trace_id):
        prompt = f"""
        You are an automated answer quality evaluator.

        STRICT RULES:
        - Return ONLY valid JSON
        - No explanations
        - No markdown
        - No extra text

        JSON SCHEMA:
        {{
        "relevance": 0-5,
        "completeness": 0-5,
        "correctness": 0-5,
        "hallucination_risk": 0-5
        }}

        Query:
        {query}

        Answer:
        {answer}
        """
        t0 = time.time()
        res = self.llm.generate_content(prompt)

        latency = time.time() - t0

        log_db(
            "llm_calls",
            ["trace_id","model","purpose","prompt","response","latency"],
            (trace_id,"gemini","critic",prompt,res.text,latency)
        )

        parsed = extract_json(res.text)

        # -------- DEFENSIVE FALLBACK --------
        if not parsed or not isinstance(parsed, dict):
            return 0.5  # neutral score

        relevance = parsed.get("relevance", 3)
        completeness = parsed.get("completeness", 3)
        correctness = parsed.get("correctness", 3)
        hallucination = parsed.get("hallucination_risk", 2)

        try:
            score = (
                (relevance + completeness + correctness) / 15
            ) * (1 - hallucination / 5)
            return round(score, 2)
        except Exception:
            return 0.5


    # ================= CACHING =================
    def _get_cache(self, query_text: str):
        import numpy as np
        q_emb = genai.embed_content(model="gemini-embedding-001", content=query_text)["embedding"]
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT query_text, query_embedding, response_json FROM semantic_cache")
        rows = cur.fetchall()
        conn.close()

        best_match = None
        max_sim = -1

        for row_text, row_emb_blob, row_res in rows:
            row_emb = np.frombuffer(row_emb_blob, dtype=np.float32)
            # Cosine similarity
            sim = np.dot(q_emb, row_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(row_emb))
            if sim > max_sim:
                max_sim = sim
                best_match = row_res

        if max_sim > self.cache_threshold:
            return json.loads(best_match)
        return None

    def _set_cache(self, query_text: str, response_data: dict):
        import numpy as np
        q_emb = genai.embed_content(model="gemini-embedding-001", content=query_text)["embedding"]
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO semantic_cache (query_text, query_embedding, response_json) VALUES (?, ?, ?)",
            (query_text, np.array(q_emb, dtype=np.float32).tobytes(), json.dumps(response_data))
        )
        conn.commit()
        conn.close()

    # ================= EXECUTION =================
    def agentic_execute(self, query, session_id="default"):
        # Check cache first
        cached = self._get_cache(query)
        if cached:
            # Update history even for cached results
            self.history.append({"role": "user", "content": query})
            self.history.append({"role": "assistant", "content": cached["answer"]})
            if len(self.history) > 10: self.history = self.history[-10:]
            return cached["answer"], cached["r_score"], cached["r_quality"], cached["a_score"]

        trace_id = str(uuid.uuid4())
        start = time.time()

        if self.df is not None:
            answer = self.pandas_agent(query, trace_id, history=self.history)
            retrieval_score, retrieval_quality = None, None
            route = "pandas"
        else:
            answer, retrieval_score, retrieval_quality = self.rag_agent(query, trace_id, history=self.history)
            route = "rag"

        answer_score = self.answer_critic(query, answer, trace_id)
        latency = time.time() - start

        log_db(
            "conversations",
            [
                "trace_id","session_id","user_query","route","final_answer",
                "retrieval_score","retrieval_quality","answer_score","latency"
            ],
            (
                trace_id, session_id, query, route, answer,
                retrieval_score, retrieval_quality,
                answer_score, latency
            ))

        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": answer})
        # Keep only last 10 messages for context
        if len(self.history) > 10:
            self.history = self.history[-10:]

        # Save to cache
        self._set_cache(query, {
            "answer": answer,
            "r_score": retrieval_score,
            "r_quality": retrieval_quality,
            "a_score": answer_score
        })

        return answer, retrieval_score, retrieval_quality, answer_score

# =====================================================
# STREAMLIT UI
# =====================================================
def main():
    st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.title("🧠 Agentic RAG")
        api_key = st.text_input("Gemini API Key", type="password")
        uploads = st.file_uploader("Upload files", accept_multiple_files=True)
        if st.button("Initialize"):
            if not api_key:
                st.error("Please provide an API key.")
            else:
                engine = UnifiedDataEngine(api_key)
                if uploads:
                    with st.spinner("Ingesting files..."):
                        engine.ingest_files(uploads)
                st.session_state.engine = engine
                st.session_state.messages = []
                st.success("Initialized")
        
        if st.button("Clear Chat"):
            if "engine" in st.session_state:
                st.session_state.engine.history = []
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "metrics" in message:
                with st.expander("Insights"):
                    cols = st.columns(2)
                    cols[0].metric("Quality", f"{int(message['metrics']['a_score']*100)}%")
                    if message['metrics']['r_quality']:
                        cols[1].metric("Retrieval", message['metrics']['r_quality'])

    if prompt := st.chat_input("Ask anything..."):
        if "engine" not in st.session_state:
            st.error("Please initialize the engine first.")
            return

        # Add user message to UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, r_score, r_quality, a_score = st.session_state.engine.agentic_execute(prompt)
                st.markdown(answer)
                
                metrics = {"a_score": a_score, "r_quality": r_quality}
                with st.expander("Insights"):
                    cols = st.columns(2)
                    cols[0].metric("Quality", f"{int(a_score*100)}%")
                    if r_quality:
                        cols[1].metric("Retrieval", r_quality)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "metrics": metrics
                })

if __name__ == "__main__":
    main()

# AIzaSyC9Yc7Y4i5fkiOFdWC_gLdG6gxW3Mngbnw

# AlzaSyDTZ5kaDPJat8bZWuyR4Rmg27u4Rv4-Q-g

# AIzaSyDTZ5kaDPJat8bZWuyR4Rmg27u4Rv4-Q-g