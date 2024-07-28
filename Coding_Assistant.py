from operator import itemgetter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
import os
api_key = os.getenv("OPENAI_API_KEY")
azure_endpoint = os.getenv("OPENAI_ENDPOINT")
api_version = os.getenv("OPENAI_API_VERSION")
model = os.getenv("OPENAI_MODEL")
deployment_name = os.getenv("OPENAI_DEPLOYMENT")
embed_model = os.getenv("OPENAI_EMBEDDING_MODEL")
embedding_deployment_name = os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME")

# Load OpenAI chat model
llm = AzureChatOpenAI(
    model= model,
    azure_deployment= deployment_name,
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
    temperature=0.2,
)

embedding = AzureOpenAIEmbeddings(
    model= embed_model,
    azure_deployment= embedding_deployment_name,
    api_key= api_key,
    azure_endpoint= azure_endpoint,
    api_version= api_version,
)
Loading some Document for Reference :
# LCEL docs
url = "https://python.langchain.com/docs/expression_language/"
loader = RecursiveUrlLoader(
    url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text
)
docs = loader.load()

# LCEL w/ PydanticOutputParser (outside the primary LCEL docs)
url = "https://python.langchain.com/docs/modules/model_io/output_parsers/quick_start"
loader = RecursiveUrlLoader(
    url=url, max_depth=1, extractor=lambda x: Soup(x, "html.parser").text
)
docs_pydantic = loader.load()

# LCEL w/ Self Query (outside the primary LCEL docs)
url = "https://python.langchain.com/docs/modules/data_connection/retrievers/self_query/"
loader = RecursiveUrlLoader(
    url=url, max_depth=1, extractor=lambda x: Soup(x, "html.parser").text
)
docs_sq = loader.load()

# Add
docs.extend([*docs_pydantic, *docs_sq])

# Sort the list based on the URLs in 'metadata' -> 'source'
d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
d_reversed = list(reversed(d_sorted))

# Concatenate the 'page_content' of each sorted dictionary
concatenated_content = "\n\n\n --- \n\n\n".join(
    [doc.page_content for doc in d_reversed]
)
from typing import TypedDict, Dict

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]
Defining the Graph nodes
def generate(state: GraphState):
    """
    Generate a code solution based on LCEL docs and the input question
    with optional feedback from code execution tests

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """

    ## State
    state_dict = state["keys"]
    question = state_dict["question"]
    iter = state_dict["iterations"]

    ## Data model
    class code(BaseModel):
        """Code output"""

        prefix: str = Field(description="Description of the problem and approach")
        imports: str = Field(description="Code block import statements")
        code: str = Field(description="Code block not including import statements")

    ## LLM
    model = llm

    # Tool
    code_tool_oai = convert_to_openai_tool(code)

    # LLM with tool and enforce invocation
    llm_with_tool = model.bind(
        tools=[code_tool_oai],
        tool_choice={"type": "function", "function": {"name": "code"}},
    )

    # Parser
    parser_tool = PydanticToolsParser(tools=[code])

    ## Prompt
    template = """You are a coding assistant with expertise in LCEL, LangChain expression language. \n
        Here is a full set of LCEL documentation:
        \n ------- \n
        {context}
        \n ------- \n
        Answer the user question based on the above provided documentation. \n
        Ensure any code you provide can be executed with all required imports and variables defined. \n
        Structure your answer with a description of the code solution. \n
        Then list the imports. And finally list the functioning code block. \n
        Here is the user question: \n --- --- --- \n {question}"""

    ## Generation
    if "error" in state_dict:
        print("---RE-GENERATE SOLUTION w/ ERROR FEEDBACK---")

        error = state_dict["error"]
        code_solution = state_dict["generation"]

        # Udpate prompt
        addendum = """  \n --- --- --- \n You previously tried to solve this problem. \n Here is your solution:
                    \n --- --- --- \n {generation}  \n --- --- --- \n  Here is the resulting error from code
                    execution:  \n --- --- --- \n {error}  \n --- --- --- \n Please re-try to answer this.
                    Structure your answer with a description of the code solution. \n Then list the imports.
                    And finally list the functioning code block. Structure your answer with a description of
                    the code solution. \n Then list the imports. And finally list the functioning code block.
                    \n Here is the user question: \n --- --- --- \n {question}"""
        template = template + addendum

        # Prompt
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question", "generation", "error"],
        )

        # Chain
        chain = (
            {
                "context": lambda _: concatenated_content,
                "question": itemgetter("question"),
                "generation": itemgetter("generation"),
                "error": itemgetter("error"),
            }
            | prompt
            | llm_with_tool
            | parser_tool
        )

        code_solution = chain.invoke(
            {"question": question, "generation": str(code_solution[0]), "error": error}
        )

    else:
        print("---GENERATE SOLUTION---")

        # Prompt
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )

        # Chain
        chain = (
            {
                "context": lambda _: concatenated_content,
                "question": itemgetter("question"),
            }
            | prompt
            | llm_with_tool
            | parser_tool
        )

        code_solution = chain.invoke({"question": question})

    iter = iter + 1
    return {
        "keys": {"generation": code_solution, "question": question, "iterations": iter}
    }
Check the import nodes
def check_code_imports(state: GraphState):
    """
    Check imports

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, error
    """

    ## State
    print("---CHECKING CODE IMPORTS---")
    state_dict = state["keys"]
    question = state_dict["question"]
    code_solution = state_dict["generation"]
    imports = code_solution[0].imports
    iter = state_dict["iterations"]

    try:
        # Attempt to execute the imports
        exec(imports)
    except Exception as e:
        print("---CODE IMPORT CHECK: FAILED---")
        # Catch any error during execution (e.g., ImportError, SyntaxError)
        error = f"Execution error: {e}"
        if "error" in state_dict:
            error_prev_runs = state_dict["error"]
            error = error_prev_runs + "\n --- Most recent run error --- \n" + error
    else:
        print("---CODE IMPORT CHECK: SUCCESS---")
        # No errors occurred
        error = "None"

    return {
        "keys": {
            "generation": code_solution,
            "question": question,
            "error": error,
            "iterations": iter,
        }
    }
Check the codes node
def check_code_execution(state: GraphState):
    """
    Check code block execution

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, error
    """

    ## State
    print("---CHECKING CODE EXECUTION---")
    state_dict = state["keys"]
    question = state_dict["question"]
    code_solution = state_dict["generation"]
    prefix = code_solution[0].prefix
    imports = code_solution[0].imports
    code = code_solution[0].code
    code_block = imports + "\n" + code
    iter = state_dict["iterations"]

    try:
        # Attempt to execute the code block
        exec(code_block)
    except Exception as e:
        print("---CODE BLOCK CHECK: FAILED---")
        # Catch any error during execution (e.g., ImportError, SyntaxError)
        error = f"Execution error: {e}"
        if "error" in state_dict:
            error_prev_runs = state_dict["error"]
            error = error_prev_runs + "\n --- Most recent run error --- \n" + error
    else:
        print("---CODE BLOCK CHECK: SUCCESS---")
        # No errors occurred
        error = "None"

    return {
        "keys": {
            "generation": code_solution,
            "question": question,
            "error": error,
            "prefix": prefix,
            "imports": imports,
            "iterations": iter,
            "code": code,
        }
    }
Execute the code node
### Edges


def decide_to_check_code_exec(state: GraphState):
    """
    Determines whether to test code execution, or re-try answer generation.

    Args:
       state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---DECIDE TO TEST CODE EXECUTION---")
    state_dict = state["keys"]
    error = state_dict["error"]

    if error == "None":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: TEST CODE EXECUTION---")
        return "check_code_execution"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: RE-TRY SOLUTION---")
        return "generate"


Define the last node
### Edges

def decide_to_finish(state: GraphState):
    """
    Determines whether to finish (re-try code 3 times.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---DECIDE TO TEST CODE EXECUTION---")
    state_dict = state["keys"]
    error = state_dict["error"]
    iter = state_dict["iterations"]

    if error == "None" or iter == 3:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: TEST CODE EXECUTION---")
        return "end"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: RE-TRY SOLUTION---")
        return "generate"
Define the Flow of the Graph and Compile the App:
from langgraph.graph import END, StateGraph

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("generate", generate)  # generation solution
workflow.add_node("check_code_imports", check_code_imports)  # check imports
workflow.add_node("check_code_execution", check_code_execution)  # check execution

# Build graph
workflow.set_entry_point("generate")
workflow.add_edge("generate", "check_code_imports")
workflow.add_conditional_edges(
    "check_code_imports",
    decide_to_check_code_exec,
    {
        "check_code_execution": "check_code_execution",
        "generate": "generate",
    },
)
workflow.add_conditional_edges(
    "check_code_execution",
    decide_to_finish,
    {
        "end": END,
        "generate": "generate",
    },
)

# Compile
app = workflow.compile()
Generate the Code Output Model:
from langchain_core.runnables import RunnableLambda


## Data model
class code(BaseModel):
    """Code output"""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")


## LLM
model = llm

# Tool
code_tool_oai = convert_to_openai_tool(code)

# LLM with tool and enforce invocation
llm_with_tool = model.bind(
    tools=[convert_to_openai_tool(code_tool_oai)],
    tool_choice={"type": "function", "function": {"name": "code"}},
)

# Parser
parser_tool = PydanticToolsParser(tools=[code])

# Create a prompt template with format instructions and the query
prompt = PromptTemplate(
    template="""You are a coding assistant with expertise in LCEL, LangChain expression language. \n
        Here is a full set of LCEL documentation:
        \n ------- \n
        {context}
        \n ------- \n
        Answer the user question based on the above provided documentation. \n
        Ensure any code you provide can be executed with all required imports and variables defined. \n
        Structure your answer with a description of the code solution. \n
        Then list the imports. And finally list the functioning code block. \n
        Here is the user question: \n --- --- --- \n {question}""",
    input_variables=["question", "context"],
)
Running Chain for Inference :
def parse_answer_to_dict(x):
    return x[0].dict()


chain_base_case = (
    {
        "context": lambda _: concatenated_content,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm_with_tool
    | parser_tool
    | RunnableLambda(parse_answer_to_dict)
)
answer = chain_base_case.invoke("How can I write a RAG chain?")
answer
