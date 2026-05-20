![Tests](https://github.com/AdithyaRaoK14/Autonomous-Data-Analyst-Agent/actions/workflows/tests.yml/badge.svg)
# 📊 Autonomous Data Analyst Agent

An AI-powered autonomous data analysis system built using LangGraph, Streamlit, and Ollama-based local LLMs.

This agent can:

* understand natural language questions
* analyze uploaded datasets
* generate Python code autonomously
* execute code securely in a sandbox
* generate visualizations
* retry after failures
* trace every reasoning step

---

# 🚀 Features

## 🧠 Autonomous AI Agent

The system uses a tool-calling AI agent capable of:

* reasoning step-by-step
* selecting tools dynamically
* generating Python analysis code
* fixing execution failures automatically
* retrying intelligently

Built using:

* LangGraph
* LangChain
* Ollama
* Qwen2.5:7B

---

## 📂 CSV Dataset Analysis

Upload any CSV dataset and ask questions like:

```text
Is there any class imbalance?
```

```text
Show a correlation heatmap.
```

```text
Plot distributions of all numeric columns.
```

```text
Which features are most correlated with the target variable?
```

```text
What preprocessing steps should I apply before ML training?
```

---

## 📊 Automatic Visualization Generation

The agent dynamically creates:

* Histograms
* Heatmaps
* Bar charts
* Scatter plots
* Distribution plots
* Correlation visualizations

Visualizations are:

* automatically generated
* saved in sandbox
* rendered inside Streamlit UI

---

## 🔒 Secure Python Sandbox

The project includes a custom secure execution engine with multiple protection layers.

### Security Features

* AST validation
* blocked dangerous imports
* restricted builtins
* subprocess isolation
* execution timeout handling
* sandboxed execution environment

Dangerous operations like:

* `os`
* `subprocess`
* `eval`
* `exec`
* filesystem access
* network access

are blocked automatically.

---

## 🔁 Intelligent Retry System

If generated code fails:

* traceback errors are analyzed
* corrected code is regenerated
* retries happen automatically
* infinite retry loops are prevented

---

## 📈 Execution Trace System

Every execution step is tracked:

* tool calls
* generated code
* retries
* outputs
* errors
* plots generated
* execution timing

This makes the agent fully transparent and debuggable.

---

## 🗄️ Smart Query Cache

Includes disk-backed caching with:

* automatic cache invalidation
* CSV modification detection
* cache bypass mode
* plot validation

Repeated questions are answered instantly.

---

# 🖼️ UI Screenshots

## Main Application UI

![Main UI](Screenshots/UI%20Image1.png)

---

## Agent Execution + Visualizations

![Execution UI](Screenshots/UI%20Image2.png)

---

# 🏗️ System Architecture

```text
User Query
    ↓
LangGraph Agent
    ↓
Tool Selection
    ↓
Python Code Generation
    ↓
Secure Sandbox Execution
    ↓
Visualization / Analysis
    ↓
Execution Trace + Final Response
```

---

# 📂 Project Structure

```text
data-analyst-agent/
│
├── agent/
│   ├── cache.py
│   ├── config.py
│   ├── graph.py
│   ├── logger.py
│   ├── prompts.py
│   ├── state.py
│   ├── tools.py
│   └── tracer.py
│
├── sandbox/
│   └── executor.py
│
├── tests/
│   ├── conftest.py
│   ├── test_agent.py
│   └── test_coverage_boost.py
│
├── Screenshots/
│   ├── UI Image1.png
│   └── UI Image2.png
│
├── app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

# ⚙️ Tech Stack

## AI / Agent Frameworks

* LangGraph
* LangChain
* Ollama

## LLM

* Qwen2.5:7B

## Frontend

* Streamlit

## Data Processing

* Pandas
* NumPy

## Visualization

* Matplotlib
* Seaborn

# 🧪 Running Tests

Run all tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=agent --cov=sandbox --cov-report=term-missing
```

Run by category:
```bash
pytest tests/ -v -k "cache"
pytest tests/ -v -k "retry"
pytest tests/ -v -k "end_to_end"
```

---

# 🔒 Security Architecture

The sandbox execution engine implements 3 security layers.

## 1. AST Validation

Blocks:

* dangerous imports
* malicious attribute access
* restricted builtins

## 2. Restricted Builtins

Only safe Python builtins are exposed.

## 3. Subprocess Isolation

Generated code executes in isolated child processes with timeout enforcement.

---

# 🧠 Agent Workflow

## Step 1 — Dataset Loading

The uploaded dataset is loaded and analyzed.

## Step 2 — Prompt Understanding

The LLM determines:

* required analysis
* necessary tools
* whether visualization is needed

## Step 3 — Code Generation

Python analysis code is generated dynamically.

## Step 4 — Secure Execution

The generated code executes inside a sandbox environment.

## Step 5 — Error Recovery

If execution fails:

* traceback is analyzed
* corrected code is generated
* retries occur automatically

## Step 6 — Final Response

Results, plots, and execution traces are displayed in the UI.

---

# 📊 Example Queries

## Dataset Overview

```text
Load the dataset and give me a complete overview.
```

## Correlation Analysis

```text
What are the top 5 features most correlated with the target variable?
```

## Visualization

```text
Show a correlation heatmap.
```

## Distribution Analysis

```text
Plot the distribution of all numeric columns.
```

## Outlier Detection

```text
Are there significant outliers in the dataset?
```

## ML Recommendations

```text
What preprocessing steps are recommended before training an ML model?
```

---

# ▶️ Local Setup

## 1. Clone Repository

```bash
git clone https://github.com/AdithyaRaoK14/Autonomous-Data-Analyst-Agent.git
cd data-analyst-agent
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Install Ollama

Download Ollama:

```text
https://ollama.com
```

Pull the required model:

```bash
ollama pull qwen2.5:7b
```

Start Ollama:

```bash
ollama serve
```

---

## 5. Run Streamlit App

```bash
streamlit run app.py
```


---

# 🐳 Docker Support

## Build Docker Image

```bash
docker build -t data-analyst-agent .
```

## Run Docker Container

```bash
docker run -p 8501:8501 data-analyst-agent
```

---



# 🧪 Test Coverage

92 tests · 86% coverage across agent and sandbox modules.

- Cache system (set, get, expiry, invalidation, file-change detection)
- AST validation and sandbox security
- Secure execution (edge cases: empty DataFrames, null columns, timeouts)
- LangGraph node unit tests (error handler, visual enforcer, plot tracker, routing)
- Retry system verification (recovery injection, hard stop at 3 failures)
- End-to-end graph integration tests with mocked LLM
- Logger structured event writing and session filtering
- Execution trace dataclass methods


---

# 📌 Engineering Concepts Demonstrated

* Agentic AI systems
* Tool-calling LLM workflows
* LangGraph state orchestration
* Secure code execution
* Autonomous retry systems
* Observability & tracing
* Streamlit application architecture
* Intelligent caching systems

---

# 🙌 Acknowledgements

Built using:

* LangChain
* LangGraph
* Ollama
* Streamlit
* Pandas
* Matplotlib
* Seaborn

---

# 📜 License

MIT License
