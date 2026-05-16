# Advanced RAG Pipeline with TruLens Evaluation
> A 4-lesson progressive project building and evaluating advanced RAG retrieval strategies — Basic RAG, Sentence Window Retrieval, Auto-Merging Retrieval, and RAG Triad Metrics — using LlamaIndex, OpenAI, and TruLens for systematic quality measurement.

---

## What This Project Does

Most RAG systems use basic chunking — split document, embed, retrieve, answer. This project goes further by implementing and **scientifically evaluating** three different retrieval strategies to find which one produces the most accurate, grounded answers.

```
PDF Document
      ↓
Three RAG Strategies Built & Compared:
  1. Basic RAG          — standard vector search baseline
  2. Sentence Window    — retrieves surrounding context per sentence
  3. Auto-Merging       — hierarchical chunks that merge when retrieved
      ↓
TruLens Evaluation Dashboard
  → Answer Relevance score
  → Context Relevance score
  → Groundedness score
      ↓
Best strategy identified with data — not guesswork
```

---

## Project Structure — 4 Lessons

---

### Lesson 1 — Advanced RAG Pipeline Overview

Builds the full pipeline from scratch and introduces TruLens evaluation.

**What's implemented:**
- Load PDF using `SimpleDirectoryReader`
- Build basic `VectorStoreIndex` with GPT-3.5-turbo
- Run queries and evaluate with TruLens leaderboard
- Compare Basic RAG vs Sentence Window vs Auto-Merging side by side

```python
# Basic pipeline
index = VectorStoreIndex.from_documents([document], service_context=service_context)
query_engine = index.as_query_engine()
response = query_engine.query("What are steps to build experience in AI?")
```

---

### Lesson 2 — RAG Triad of Metrics

Implements the 3 core evaluation metrics that measure RAG quality scientifically.

**The RAG Triad:**

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **Answer Relevance** | Does the answer address the question? | Catches hallucinations |
| **Context Relevance** | Is retrieved context actually relevant? | Catches noisy retrieval |
| **Groundedness** | Is the answer supported by retrieved context? | Catches made-up facts |

```python
# Answer Relevance
f_qa_relevance = Feedback(provider.relevance_with_cot_reasons,
    name="Answer Relevance").on_input_output()

# Context Relevance
f_qs_relevance = Feedback(provider.qs_relevance_with_cot_reasons,
    name="Context Relevance").on_input().on(context_selection).aggregate(np.mean)

# Groundedness
f_groundedness = Feedback(grounded.groundedness_measure_with_cot_reasons,
    name="Groundedness").on(context_selection).on_output()
```

---

### Lesson 3 — Sentence Window Retrieval

Solves the problem of losing context when chunks are too small.

**The problem with basic RAG:**
```
Basic RAG retrieves: "Networking is important."
→ Too little context — LLM can't give a useful answer
```

**Sentence Window solution:**
```
Retrieve: "Networking is important."
Return window: "[3 sentences before] Networking is important. [3 sentences after]"
→ LLM has full context → better answer
```

**How it works:**
```python
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,                          # 3 sentences each side
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)
# After retrieval — replace small chunk with full window
postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
# Rerank retrieved nodes for highest relevance
rerank = SentenceTransformerRerank(top_n=2, model="BAAI/bge-reranker-base")
```

**Evaluated at window sizes 1 and 3 — TruLens scores compared.**

---

### Lesson 4 — Auto-Merging Retrieval

Solves the problem of retrieving fragmented chunks from the same parent section.

**The problem:**
```
Many small chunks retrieved → all from same paragraph
→ Redundant, wastes context window
```

**Auto-Merging solution:**
```
Hierarchical chunks: [2048] → [512] → [128]
If enough small chunks from same parent retrieved
→ Automatically merge into parent chunk
→ One coherent section instead of fragments
```

**How it works:**
```python
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]   # parent → child → grandchild
)
retriever = AutoMergingRetriever(
    base_retriever,
    automerging_index.storage_context,
    verbose=True
)
```

**Evaluated at 2-layer [2048, 512] and 3-layer [2048, 512, 128] — TruLens leaderboard compares all.**

---

## Architecture Overview

```
PDF Document (eBook: How to Build a Career in AI)
         ↓
SimpleDirectoryReader → Document object
         ↓
Three Indexing Strategies:

[Strategy 1]              [Strategy 2]              [Strategy 3]
Basic RAG                 Sentence Window           Auto-Merging
VectorStoreIndex          SentenceWindowNode        HierarchicalNode
standard chunks           Parser + window           Parser + leaf nodes
                          metadata                  + storage context
         ↓                        ↓                        ↓
         └──────────────┬─────────────────────────────────┘
                        ↓
              TruLens Evaluation
              ┌─────────────────┐
              │ Answer Relevance│
              │ Context Relev.  │
              │ Groundedness    │
              └─────────────────┘
                        ↓
              TruLens Dashboard
              Leaderboard comparison
              Best strategy identified
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-RAG%20Framework-purple?style=flat-square)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5--turbo-black?style=flat-square)
![TruLens](https://img.shields.io/badge/TruLens-RAG%20Evaluation-green?style=flat-square)

- **LlamaIndex** — document loading, indexing, node parsing, query engines
- **OpenAI GPT-3.5-turbo** — language model for generation
- **BAAI/bge-small-en-v1.5** — local embedding model (HuggingFace)
- **BAAI/bge-reranker-base** — sentence transformer reranker
- **TruLens** — RAG evaluation framework with Answer Relevance, Context Relevance, Groundedness
- **Python** — core language

---

## Project Structure

```
Advanced-RAG-TruLens/
│
├── lesson1_advanced_rag_pipeline.ipynb    # Full pipeline + TruLens intro
├── lesson2_rag_triad_metrics.ipynb        # 3 evaluation metrics implementation
├── lesson3_sentence_window_retrieval.ipynb # Sentence window + reranking
├── lesson4_auto_merging_retrieval.ipynb   # Hierarchical auto-merging
│
├── utils.py                               # Helper functions
├── eBook-How-to-Build-a-Career-in-AI.pdf  # Source document
├── eval_questions.txt                     # Evaluation question set
├── generated_questions.text               # Auto-generated eval questions
│
├── sentence_index/                        # Persisted sentence window index
├── merging_index/                         # Persisted auto-merging index
│
├── requirements.txt                       # All dependencies
├── .env.example                           # API key template
└── README.md
```

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Saqib00712/IBM_RAG_Specialization.git
cd IBM_RAG_Specialization
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up API key
```bash
cp .env.example .env
```
Edit `.env`:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run lessons in order
```bash
jupyter notebook lesson1_advanced_rag_pipeline.ipynb
```

> Run each lesson notebook in order — each builds on concepts from the previous one.

---

## Key Concepts Covered

- **Basic RAG pipeline** — `SimpleDirectoryReader`, `VectorStoreIndex`, `ServiceContext`
- **Sentence Window Retrieval** — `SentenceWindowNodeParser`, `MetadataReplacementPostProcessor`
- **Auto-Merging Retrieval** — `HierarchicalNodeParser`, `AutoMergingRetriever`, `StorageContext`
- **Reranking** — `SentenceTransformerRerank` with `BAAI/bge-reranker-base`
- **RAG Triad evaluation** — Answer Relevance, Context Relevance, Groundedness
- **TruLens integration** — `TruLlama`, `FeedbackMode`, `Tru().get_leaderboard()`
- **Index persistence** — saving and loading indexes from disk
- **Systematic comparison** — comparing strategies with quantitative metrics

---

## Example Queries Used for Evaluation

```python
eval_questions = [
    "What are steps to take when finding projects to build your experience?",
    "How do you create your AI portfolio?",
    "What is the right AI job for me?",
    "How can I be successful in AI?",
    "What is the importance of networking in AI?",
    "How do I build a portfolio of AI projects?",
]
```

---

## TruLens Dashboard

```python
# Launch evaluation dashboard
tru.run_dashboard()  # opens at http://localhost:8501/

# Compare all strategies on leaderboard
tru.get_leaderboard(app_ids=[])
```

The dashboard shows a ranked comparison of:
- Direct Query Engine (Basic RAG)
- Sentence Window Engine (window=1)
- Sentence Window Engine (window=3)
- Auto-Merging Engine (2 layers)
- Auto-Merging Engine (3 layers)

---

## Course Credit

This project was built following the **Building and Evaluating Advanced RAG Applications** short course on [DeepLearning.AI](https://www.deeplearning.ai), created in collaboration with LlamaIndex and TruEra. The implementation and documentation were extended and personalized independently.

---

## Related Certifications

Built as part of the IBM **RAG for Generative AI Applications Specialization** and **AI Agents Using RAG and LangChain** on Coursera.

[![IBM Badge](https://img.shields.io/badge/IBM-RAG%20Specialization-blue?style=flat-square)](https://www.credly.com/users/muhammad-saqib.361f9b8c)
[![IBM Badge](https://img.shields.io/badge/IBM-AI%20Agents%20%26%20RAG-blue?style=flat-square)](https://www.credly.com/users/muhammad-saqib.361f9b8c)

---

## Author

**Muhammad Saqib**
- GitHub: [@Saqib00712](https://github.com/Saqib00712)
- LinkedIn: [muhammad-saqib](https://www.linkedin.com/in/muhammad-saqib-68b9b3374/)
- Email: saqibkhosa649@gmail.com
- Credly: [15x IBM Certified](https://www.credly.com/users/muhammad-saqib.361f9b8c)
