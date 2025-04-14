# 🧠 Talk to DB Using LangGraph

A full-stack conversational assistant that allows users to query a PostgreSQL database using natural language. It leverages LangGraph + LangChain on the backend and a custom-built, responsive chat UI on the frontend.

> Built for exploring NPS feedback insights, but easily adaptable to any structured dataset.

---

## 📸 Demo

![demo](assets/demo.gif)

---

## 🧰 Tech Stack

| Layer     | Stack                                               |
|-----------|-----------------------------------------------------|
| Frontend  | HTML + CSS + Vanilla JavaScript                     |
| Backend   | FastAPI + LangGraph + LangChain + OpenAI GPT-4o     |
| Database  | PostgreSQL (via SQLAlchemy connection)              |

---

## 🗂️ Project Structure

```
.
├── main.py               # FastAPI backend with LangGraph workflow
├── templates/
│   └── index.html        # Frontend chat UI
├── .env                  # API key config (Add:OPENAI_API_KEY="your key" format)
├── requirements.txt      # Python dependencies
└── README.md             # Project docs
```

---

## ⚙️ Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/arun1934/TalktoDBUsingLangGraph.git
cd TalktoDBUsingLangGraph
```

### 2. Setup Python Environment

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_key_here
```

Update the DB URI in `main.py` if needed:

```python
db = SQLDatabase.from_uri("postgresql+psycopg2://postgres:admin@localhost/NPS")
```

---

### 4. Start the App

```bash
python main.py
```

Visit: [http://localhost:8003](http://localhost:8003)

---

## 🔍 Implementation Details

---

### 🧠 Backend (LangGraph + FastAPI)

#### 🏗️ Workflow Overview

```python
workflow = StateGraph(State)

workflow.add_node("first_tool_call", first_tool_call)
workflow.add_node("list_tables_tool", create_tool_node_with_fallback([list_tables_tool]))
workflow.add_node("get_schema_tool", create_tool_node_with_fallback([get_schema_tool]))
workflow.add_node("model_get_schema", log_schema)
workflow.add_node("query_gen", query_gen_node)
workflow.add_node("correct_query", model_check_query)
workflow.add_node("execute_query", create_tool_node_with_fallback([db_query_tool]))
workflow.add_node("query_gen_node_for_formatting", query_gen_node_for_formatting)
workflow.add_node("generate_suggestions", generate_suggestions_node)
```

#### ⚙️ Natural Language → SQL Prompt

```python
query_gen_system = """
You are an agent designed to interact with a SQL database...
- Never assume columns or tables not shown in schema.
- Always validate queries before execution.
- DO NOT use INSERT, UPDATE, DELETE, DROP.
"""
```

#### ✅ Query Execution & Fallback

```python
@tool(name_or_callable="db_query_tool")
def db_query_tool(query: str) -> dict:
    result_str = db.run_no_throw(query)
    result = eval(result_str, {"Decimal": Decimal})
    rows = [dict(zip(columns, row)) for row in result]
    return {"columns": columns, "rows": rows}
```

---

### 💬 Frontend (Pure HTML + JS)

#### 🖼️ Chat Bubbles + Feedback UI

```html
<div class="message bot-message">
  <div class="timestamp">09:30</div>
  <div class="feedback-buttons">
    <span onclick="handleFeedback(true, this)">👍</span>
    <span onclick="handleFeedback(false, this)">👎</span>
  </div>
</div>
```

#### 🔁 Retry Logic on Dislike

```javascript
function handleFeedback(isLike, element) {
  if (!isLike) {
    const line = document.createElement("div");
    line.innerHTML = "🔁 Reworking for a better answer<span class='dot-anim'></span>";
    userInput.value = query;
    sendMessage(true, msg);
  }
}
```

#### 📤 Sending User Query

```javascript
async function sendMessage(isRetry = false) {
  const response = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query })
  });
  const data = await response.json();
  appendMessage(data.explanation, false, { query });
}
```

---

## ✨ Example Prompts

```text
What are the top comments received last month?
Compare themes between January and February.
Run sentiment analysis and suggest 3 improvements.
Which delivery partner has lowest satisfaction?
```

---

## 🧪 Debugging Tips

- Use `set_verbose(True)` and `print()` to log agent behavior
- Check console logs for:
  - 📐 Retrieved schema
  - 🧠 Final SQL
  - 🔁 Event update

---

## 📊 Future Enhancements

- [ ] Visual charts based on query results (e.g. bar/pie)
- [ ] CSV export of results
- [ ] Secure login + multi-tenant access
- [ ] Admin dashboard for query history
- [ ] Deploy via Docker or Render

---

## 🤝 Contributing

Feel free to fork this project and contribute via PRs or issues.

---

## 📄 License

MIT License

---

## 🙏 Acknowledgements

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [LangChain](https://github.com/langchain-ai/langchain)
- [OpenAI](https://openai.com)
- [FastAPI](https://fastapi.tiangolo.com)
- [LangChain Tutorial] (https://langchain-ai.github.io/langgraph/tutorials/sql-agent/)
