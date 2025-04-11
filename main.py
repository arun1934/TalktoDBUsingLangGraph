from dotenv import load_dotenv
from typing import Annotated, Any, Literal
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.tools import tool
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import os
import json
import re

# --- FastAPI app ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- Environment and DB Setup ---
load_dotenv()
os.getenv("OPENAI_API_KEY")
db = SQLDatabase.from_uri("postgresql+psycopg2://postgres:admin@localhost/NPS")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

# --- Tool + Error Handling ---
def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    return ToolNode(tools).with_fallbacks([RunnableLambda(handle_tool_error)], exception_key="error")

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

# --- Tooling ---
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")

@tool
def db_query_tool(query: str) -> dict:
    """Execute a SQL query and return results as structured JSON."""
    result = db.run_no_throw(query)
    if not result:
        return {"error": "Query failed or returned no data."}
    rows = [dict(row) for row in result]
    columns = list(rows[0].keys()) if rows else []
    return {"columns": columns, "rows": rows}

# --- LangGraph State ---
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

workflow = StateGraph(State)

# --- Initial Tool Invocation ---
def first_tool_call(state: State) -> dict[str, list[AIMessage]]:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sql_db_list_tables",
                        "args": {},
                        "id": "tool_ai",
                    }
                ],
            )
        ]
    }

# --- Query Correction Prompt & Node ---
query_check_system = """You are a SQL expert with a strong attention to detail.
Double check the SQLite query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

You will call the appropriate tool to execute the query after running this check."""

query_check_prompt = ChatPromptTemplate.from_messages([
    ("system", query_check_system), ("placeholder", "{messages}")
])
query_check = query_check_prompt | llm.bind_tools(
    [db_query_tool], tool_choice="required"
)

def model_check_query(state: State) -> dict[str, list[AIMessage]]:
    return {"messages": [query_check.invoke({"messages": [state["messages"][-1]]})]}


# --- Schema Inspection ---
model_get_schema = llm.bind_tools([get_schema_tool])

# --- Query Generation Prompt ---
class SubmitFinalAnswer(BaseModel):
    final_answer: str = Field(..., description="The final answer to the user")


query_gen_system = """You are a SQL expert with a strong attention to detail.

Given an input question, output a syntactically correct SQLite query to run, then look at the results of the query and return the answer.

DO NOT call any tool besides SubmitFinalAnswer to submit the final answer.

When generating the query:

Output the SQL query that answers the input question without a tool call.

Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.

If you get an error while executing a query, rewrite the query and try again.
If you get a SELECT query as the final answer, rewrite the query and try again.

If you get an empty result set, you should try to rewrite the query to get a non-empty result set.
NEVER make stuff up if you don't have enough information to answer the query... just say you don't have enough information.

If you have enough information to answer the input question, simply invoke the appropriate tool to submit the final answer to the user.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database."""

query_gen_prompt = ChatPromptTemplate.from_messages([
    ("system", query_gen_system), ("placeholder", "{messages}")
])
query_gen = query_gen_prompt | llm.bind_tools([SubmitFinalAnswer])

def query_gen_node(state: State):
    message = query_gen.invoke(state)
    tool_messages = []
    if message.tool_calls:
        for tc in message.tool_calls:
            if tc["name"] != "SubmitFinalAnswer":
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: The wrong tool was called: {tc['name']}...",
                        tool_call_id=tc["id"],
                    )
                )
    return {"messages": [message] + tool_messages}


# --- Suggestion Generator ---
suggestion_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an assistant that helps users explore their database through natural language.
Given a user query and the resulting data, suggest 3 follow-up questions.
The three questions should be relevant to the data returned and the original query.
Make sure there is no repeat of these suggestions.
Only return a JSON list of strings like:
["Follow-up 1", "Follow-up 2", "Follow-up 3"]
"""),
    ("user", "Query: {query}\nResult: {result}")
])
suggestion_chain = suggestion_prompt | llm

def generate_suggestions_node(state: State) -> dict:
    user_query = ""
    result = ""
    print("🧠 Gathering context for suggestions...")
    for msg in state["messages"]:
        if getattr(msg, "type", "") == "human":
            user_query = msg.content
        elif isinstance(msg, ToolMessage) and "Error" not in msg.content:
            result = msg.content
        elif isinstance(msg, AIMessage) and "SubmitFinalAnswer" in str(msg.tool_calls):
            for call in msg.tool_calls:
                result = call.get("args", {}).get("final_answer", "")
    try:
        print("📤 Suggestion Input - Query:", user_query)
        print("📤 Suggestion Input - Result:", result)
        response = suggestion_chain.invoke({"query": user_query, "result": result})
        print("📥 Raw Suggestion Output:", response.content)
        suggestions = json.loads(response.content)
        print("✅ Suggestions:123", suggestions)
        state["messages"].append(
            ToolMessage(
                content=suggestions,
                tool_call_id="suggestion_tool",
            )
        )
    except:
        suggestions = []
    return {"messages": state["messages"], "suggestions": suggestions}

def extract_suggestions(event_result):
    messages = event_result.get("generate_suggestions", {}).get("messages", [])
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.tool_call_id == "suggestion_tool":
            try:
                return msg.content
            except Exception as e:
                print(f"Error parsing suggestions: {e}")
                return []
    return []

def should_continue(state: State) -> Literal[END, "correct_query", "query_gen", "generate_suggestions"]:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "generate_suggestions"
    if last_message.content.startswith("Error:"):
        return "query_gen"
    else:
        return "correct_query"

# --- Workflow Nodes ---
workflow.add_node("first_tool_call", first_tool_call)
workflow.add_node("list_tables_tool", create_tool_node_with_fallback([list_tables_tool]))
workflow.add_node("get_schema_tool", create_tool_node_with_fallback([get_schema_tool]))
workflow.add_node("model_get_schema", lambda state: {"messages": [model_get_schema.invoke(state["messages"])]})
workflow.add_node("query_gen", query_gen_node)
workflow.add_node("correct_query", model_check_query)
workflow.add_node("execute_query", create_tool_node_with_fallback([db_query_tool]))
workflow.add_node("generate_suggestions", generate_suggestions_node)

# --- Workflow Edges ---
workflow.add_edge(START, "first_tool_call")
workflow.add_edge("first_tool_call", "list_tables_tool")
workflow.add_edge("list_tables_tool", "model_get_schema")
workflow.add_edge("model_get_schema", "get_schema_tool")
workflow.add_edge("get_schema_tool", "query_gen")
workflow.add_conditional_edges("query_gen", should_continue)
workflow.add_edge("correct_query", "execute_query")
workflow.add_edge("execute_query", "query_gen")
workflow.add_edge("generate_suggestions", END)

# --- FastAPI Routes ---
app1 = workflow.compile()

class QueryRequest(BaseModel):
    query: str

@app.post("/api/query")
async def process_query(request: QueryRequest):
    print("📥 Received query:", request.query)
    try:
        event_result = {}
        for event in app1.stream({"messages": [("user", request.query)]}):
            print("🔁 Event update:", event)
            event_result.update(event)

        result_obj = get_final_answer(event_result)
        if not result_obj:
            result_obj = {
                "sql": "",
                "explanation": "Sorry, no answer could be generated.",
                "columns": [],
                "rows": []
            }

        suggestions = extract_suggestions(event_result)
        print("💡 Suggestions:", suggestions)

        return JSONResponse(content={
            "sql": result_obj.get("sql", ""),
            "explanation": result_obj.get("explanation", ""),
            "results": {
                "columns": result_obj.get("columns", []),
                "rows": result_obj.get("rows", [])
            },
            "rowCount": len(result_obj.get("rows", [])),
            "suggestions": suggestions
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- Result Parsing ---
def extract_sql_and_explanation(text: str) -> tuple[str, str]:
    sql_block = re.search(r"```sql\s+(.*?)```", text, re.DOTALL)
    if sql_block:
        sql_code = sql_block.group(1).strip()
        explanation = re.sub(r"```sql\s+.*?```", "", text, flags=re.DOTALL).strip()
        return sql_code, explanation
    return "", text

def get_final_answer(data):
    print("📦 Extracting final answer...")
    try:
        sql_query = ""
        explanation = ""
        columns = []
        rows = []

        messages = data.get("query_gen", {}).get("messages", [])
        print("📄 Query Gen Messages:", messages)
        for message in messages:
            tool_calls = getattr(message, "tool_calls", [])
            for call in tool_calls:
                args = call.get("args", {})
                final_answer = args.get("final_answer", "")
                if final_answer:
                    sql_match = re.search(r"```sql\\s+(.*?)```", final_answer, re.DOTALL)
                    if sql_match:
                        sql_query = sql_match.group(1).strip()
                        explanation = re.sub(r"```sql\\s+.*?```", "", final_answer, flags=re.DOTALL).strip()
                    else:
                        explanation = final_answer.strip()

        exec_msgs = data.get("execute_query", {}).get("messages", [])
        print("📊 Execute Query Messages:", exec_msgs)
        for msg in exec_msgs:
            if isinstance(msg, ToolMessage):
                try:
                    result_data = json.loads(msg.content)
                    columns = result_data.get("columns", [])
                    rows = result_data.get("rows", [])
                except Exception as e:
                    print(f"Error parsing JSON from db tool message: {e}")
                    continue

        print("🧠 Final SQL:", sql_query)
        print("🗒️ Explanation:", explanation)
        print("📊 Columns:", columns)
        print("📈 Rows returned:", len(rows))
        return {
            "sql": sql_query,
            "explanation": explanation,
            "columns": columns,
            "rows": rows
        }

    except Exception as e:
        print(f"Error extracting final answer: {e}")
        return {
            "sql": "",
            "explanation": "Sorry, something went wrong.",
            "columns": [],
            "rows": []
        }

# --- Entrypoint ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
