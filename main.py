# This is a FastAPI application that serves as an agent for querying a SQL database using natural language.
# It utilizes LangChain and LangGraph to create a workflow that interacts with the database.
# At a high level, the agent will:
# 1. Fetch the available tables from the database
# 2. Decide which tables are relevant to the question
# 3. Fetch the DDL for the relevant tables
# 4. Generate a query based on the question and information from the DDL
# 5. Double-check the query for common mistakes using an LLM
# 6. Execute the query and return the results
# 7. Correct mistakes surfaced by the database engine until the query is successful
# 8. Formulate a response based on the results
# 9. Suggest follow-up questions based on the results
# 10. Provide a web interface for users to interact with the agent
# 11. Handle errors gracefully and provide feedback to the user


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
from langchain.globals import set_verbose
from langchain.globals import set_debug
from decimal import Decimal


import os
import json
import re

set_debug(False)
set_verbose(True)

# --- FastAPI app ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- Environment and DB Setup ---
load_dotenv()
os.getenv("OPENAI_API_KEY")
db = SQLDatabase.from_uri("postgresql+psycopg2://postgres:admin@localhost/NPS")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# Tools creation with feallback and error hanling
def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """Create a ToolNode with a fallback to handle errors and surface them to the agent."""
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error")


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


# Database specific tools
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")


# Tool definition for executing SQL queries
@tool(name_or_callable="db_query_tool")
def db_query_tool(query: str) -> dict:
    """
    Execute a SQL query against the database and get back the result.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """
    result = {}
    result_str = ""
    try:
        # print("Query executed in DB:", query)
        result_str = db.run_no_throw(query)
        print("Result post running query is: ", result_str)
        if not result_str:
            return {"error": "Query failed or returned no data."}
        try:
            # Attempt to convert string representation of tuples into actual tuples
            result = eval(result_str, {"Decimal": Decimal})

            if isinstance(result, (list, tuple)) and all(isinstance(row, (list, tuple)) for row in result):
                if result and all(len(row) == len(result[0]) for row in result):
                    columns = [f"column_{i + 1}" for i in range(len(result[0]))]
                    rows = [dict(zip(columns, row)) for row in result]
                    return {"result_str": result_str, "columns": columns, "rows": rows}
                else:
                    return {"error": "Inconsistent data format in result."}
            else:
                return {"error": "Unexpected result type from database. Expected a list/tuple of lists/tuples"}

        except (SyntaxError, NameError, TypeError) as eval_err:
            return {"error": f"Error converting result string to data structure: {eval_err}"}

    except Exception as e:
        print(f"❌ Exception while executing DB query: {e}")
    return {"result_str": result_str, "result": result}


# --- Query Correction Prompt & Node ---
query_check_system = """You are a SQL expert with a strong attention to detail.
Double check the SQL query for common mistakes, including:
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


# LangGraph specific code
# Define the state for the agent
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# Define a new graph
workflow = StateGraph(State)


# Add a node for the first tool call
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


def model_check_query(state: State) -> dict[str, list[AIMessage]]:
    """This tool will double check the query before executing it."""
    return {"messages": [query_check.invoke({"messages": [state["messages"][-1]]})]}


# --- Schema Inspection ---
model_get_schema = llm.bind_tools([get_schema_tool])

# --- Query Generation Prompt ---
class SubmitFinalAnswer(BaseModel):
    """Submit the final answer to to user based on the query results"""
    final_answer: str = Field(..., description="The final answer to the user")


query_gen_system = """
ROLE:
You are an agent designed to interact with a SQL database. You have access to tools for interacting with the database.
GOAL:
Given an input question, create a syntactically correct SQL query to run, then look at the results of the query and return the answer.
INSTRUCTIONS:
- Only use the below tools for the following operations.
- Never assume columns or tables that are not explicitly shown in the schema.
- Only use the information returned by the below tools to construct your final answer.
- To start you should ALWAYS look at the tables in the database to see what you can query. Do NOT skip this step.
- Then you should query the schema of the most relevant tables.
- Write your query based upon the schema of the tables. You MUST double check your query before executing it.
- Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 10 results.
- You can order the results by a relevant column to return the most interesting examples in the database.
- Never query for all the columns from a specific table, only ask for the relevant columns given the question.
- If you get an error while executing a query, rewrite the query and try again.
- If the query returns a result, use check_result tool to check the query result.
- If the query result result is empty, think about the table schema, rewrite the query, and try again.
- DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database."""

query_gen_prompt = ChatPromptTemplate.from_messages([
    ("system", query_gen_system), ("placeholder", "{messages}")
])

query_gen = query_gen_prompt | llm


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


query_gen_formatting_prompt = ChatPromptTemplate.from_messages([
    ("system", query_gen_system), ("placeholder", "{messages}")
])
# query_gen = query_gen_prompt | llm.bind_tools([SubmitFinalAnswer, model_check_query])
# query_gen = query_gen_prompt | llm.bind_tools([SubmitFinalAnswer])
query_gen_formatting_prompt = query_gen_formatting_prompt | llm.bind_tools([SubmitFinalAnswer])


def query_gen_node_for_formatting(state: State):
    message = query_gen_formatting_prompt.invoke(state)
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


# --- Suggestion Generation and related code ---
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
    # print("🧠 Gathering context for suggestions...")
    for msg in state["messages"]:
        if getattr(msg, "type", "") == "human":
            user_query = msg.content
        elif isinstance(msg, ToolMessage) and "Error" not in msg.content:
            result = msg.content
        elif isinstance(msg, AIMessage) and "SubmitFinalAnswer" in str(msg.tool_calls):
            for call in msg.tool_calls:
                result = call.get("args", {}).get("final_answer", "")
    try:
        # print("Suggestion Input - Query:", user_query)
        # print("Suggestion Input - Result:", result)
        response = suggestion_chain.invoke({"query": user_query, "result": result})
        print("Suggestion Output:", response.content)
        suggestions = json.loads(response.content)
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


# Define a conditional edge to decide whether to continue or end the workflow
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
workflow.add_node("query_gen_node_for_formatting", query_gen_node_for_formatting)

# --- Workflow Edges ---
workflow.add_edge(START, "first_tool_call")
workflow.add_edge("first_tool_call", "list_tables_tool")
workflow.add_edge("list_tables_tool", "model_get_schema")
workflow.add_edge("model_get_schema", "get_schema_tool")
workflow.add_edge("get_schema_tool", "query_gen")

workflow.add_conditional_edges("query_gen", should_continue)
workflow.add_edge("query_gen", "correct_query")
workflow.add_edge("correct_query", "execute_query")
workflow.add_edge("execute_query", "query_gen_node_for_formatting")
workflow.add_edge("query_gen_node_for_formatting", "generate_suggestions")
workflow.add_edge("generate_suggestions", END)
# --- FastAPI Routes ---
app1 = workflow.compile()

class QueryRequest(BaseModel):
    query: str

@app.post("/api/query")
async def process_query(request: QueryRequest):
    # print("Received query:", request.query)
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

        messages = data.get("correct_query", {}).get("messages", [])
        print("📄 Query Gen Messages:", messages)
        for message in messages:
            tool_calls = getattr(message, "tool_calls", [])
            for call in tool_calls:
                args = call.get("args", {})
                sql_query = args.get("query", "")

        exec_msgs = data.get("query_gen_node_for_formatting", {}).get("messages", [])
        print("📊 Execute Query Messages:", exec_msgs)
        for msg in exec_msgs:
            if isinstance(msg, AIMessage):
                try:
                    explanation =  msg.tool_calls[0].get("args").get("final_answer")
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
            "explanation": "Sorry, something went wrong. Please click on 👎 to re-try",
            "columns": [],
            "rows": []
        }

# --- Entrypoint ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
