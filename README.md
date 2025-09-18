# LGU Project

이 git은 strands SDK 기반의 LGU 헤커톤을 위해 준비되었습니다.


# LGM Project

여기서는 LangGraph 방식의 Agent를 MCP를 이용해 활용하는 방법에 대해 설명합니다. 전체적인 architecture는 아래와 같습니다. 개발의 편의를 위하여 [Streamlit](https://streamlit.io/)을 이용해 UI를 구성하고 Host는 MCP Client를 포함하고 있으며, custom MCP 서버들을 정의하여 활용합니다. AWS의 완전관리형 RAG 서비스인 Knowledge Base를 [kb-retriever](./application/mcp_server_retrieve.py)를 이용해 활용합니다. 각종 데이터의 분석을 수행하는 Code Interpreter는 [repl-coder](./application/mcp_server_repl_coder.py)을 이용합니다. 또한 AWS의 각종 리소스를 AWS CLI 기반으로 관리할 수 있는 [use-aws](./application/mcp_server_use_aws.py)을 이용해 MCP로 편리하게 이용할 수 있습니다.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/509af9b8-a045-4bdf-8cec-d8f128e05897" />

## MCP와 LangGraph를 이용하여 Agent 구현하기

아래에서는 MCP와 LangGraph를 이용해 RAG 등을 활용할 수 있는 Agent를 구현하는 방법에 대해 설명합니다.

### Agent

아래와 같이 ReAct 형태의 LangGraph Agent를 정의합니다. 여기서는 checkpoint와 memorystore를 이용해 최근 대화 내용을 참조합니다.

```python
def buildChatAgentWithHistory(tools):
    tool_node = ToolNode(tools)

    workflow = StateGraph(State)

    workflow.add_node("agent", call_model)
    workflow.add_node("action", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": END,
        },
    )
    workflow.add_edge("action", "agent")

    return workflow.compile(
        checkpointer=chat.checkpointer,
        store=chat.memorystore
    )
```

LangGraph에서는 State를 이용해 각 node들의 정보를 업데이트 합니다. 아래와 같이 add_messages를 이용하면 편리하게 messages에 각 node의 상태를 저장할 수 있습니다.

```python
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
    image_url: list
```

Agent의 동작을 수행하는 call_model은 아래와 같습니다. 별도로 system_prompt가 지정되지 않으면 아래와 같은 기본 prompt를 활용합니다. chat model은 아래와 같이 MCP에서 얻은 tool정보와 bind를 수행합니다. 이후 prompt와 model을 chain으로 묶은후 invoke를 수횅합니다. 

```python
async def call_model(state: State, config):
    last_message = state['messages'][-1]
    image_url = state['image_url'] if 'image_url' in state else []

    tools = config.get("configurable", {}).get("tools", None)
    system_prompt = config.get("configurable", {}).get("system_prompt", None)
    
    if system_prompt:
        system = system_prompt
    else:
        system = (
            "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다."
            "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
            "모르는 질문을 받으면 솔직히 모른다고 말합니다."
            "한국어로 답변하세요."

            "An agent orchestrates the following workflow:"
            "1. Receives user input"
            "2. Processes the input using a language model"
            "3. Decides whether to use tools to gather information or perform actions"
            "4. Executes those tools and receives results"
            "5. Continues reasoning with the new information"
            "6. Produces a final response"
        )

    chatModel = chat.get_chat(extended_thinking=reasoning_mode)
    model = chatModel.bind_tools(tools)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | model
        
    response = await chain.ainvoke(state["messages"])
    return {"messages": [response], "image_url": image_url}
```

Agent의 conditional edge인 should_continue는 아래와 같이 정의합니다. 

```python
async def should_continue(state: State, config) -> Literal["continue", "end"]:
    messages = state["messages"]    
    last_message = messages[-1]
    
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "continue"
    else:
        return "end"
```

### MCP의 활용

MCP에 대한 서버 정보는 [mcp_config.py](./application/mcp_config.py)에서 정의합니다. RAG를 조회하는 MCP 서버는 아래와 같이 [mcp_server_retrieve.py](./application/mcp_server_retrieve.py)로 정의하고 있습니다.

```java
{
    "mcpServers": {
        "kb_retriever": {
            "command": "python",
            "args": [f"{workingDir}/mcp_server_retrieve.py"]
        }
    }
}
```

[mcp_server_retrieve.py](./application/mcp_server_retrieve.py)에서는 FastMCP를 이용해 MCP 서버를 정의합니다. 여기서 MCP tool은 @mcp.tool() decorator를 이용해 구현하고 doc string에 tool에 대한 정보를 활용하여 적절한 tool이 선택되게 됩니다. Custom MCP를 local에서 정의해서 사용하는 경우에는 아래와 같이 transport로 stdio를 활용합니다. 


```python
import mcp_retrieve
from mcp.server.fastmcp import FastMCP 

mcp = FastMCP(
    name = "mcp-retrieve",
    instructions=(
        "You are a helpful assistant. "
        "You retrieve documents in RAG."
    ),
)

@mcp.tool()
def retrieve(keyword: str) -> str:
    """
    Query the keyword using RAG based on the knowledge base.
    keyword: the keyword to query
    return: the result of query
    """
    return mcp_retrieve.retrieve(keyword)

if __name__ =="__main__":
    mcp.run(transport="stdio")
```

RAG를 검색하는 retrieve tool은 [mcp_retrieve.py](./application/mcp_retrieve.py)와 같이 boto3의 bedrock_agent_runtime_client를 활용하여 Knowledge Base를 직접 검색합니다. 결과를 활용하기 쉬운 json 형태로 변환한 후에 리턴하면 Agent에서 활용할 수 있습니다.

```python
def retrieve(query):
    response = bedrock_agent_runtime_client.retrieve(
        retrievalQuery={"text": query},
        knowledgeBaseId=knowledge_base_id,
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": number_of_results},
            },
        )    
    retrieval_results = response.get("retrievalResults", [])

    json_docs = []
    for result in retrieval_results:
        text = url = name = None
        if "content" in result:
            content = result["content"]
            if "text" in content:
                text = content["text"]

        if "location" in result:
            location = result["location"]
            if "s3Location" in location:
                uri = location["s3Location"]["uri"] if location["s3Location"]["uri"] is not None else ""                
                name = uri.split("/")[-1]
                
            elif "webLocation" in location:
                url = location["webLocation"]["url"] if location["webLocation"]["url"] is not None else ""
                name = "WEB"

        json_docs.append({
            "contents": text,              
            "reference": {
                "url": url,                   
                "title": name,
                "from": "RAG"
            }
        })

    return json.dumps(json_docs, ensure_ascii=False)
```


### MCP Agent의 동작

[chat.py](./application/chat.py)와 같이 MCP 서버 정보를 이용하여 MCP client를 정의합니다. 이후 tool에 대한 정보를 추출합니다. 이후 아래와 같이 Agent를 생성하고, config와 함께 실행합니다. 이후 결과는 stream 형태에서 필요한 정보를 추출하여 활용합니다.

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

mcp_json = mcp_config.load_selected_config(mcp_servers)
server_params = langgraph_agent.load_multiple_mcp_server_parameters(mcp_json)

client = MultiServerMCPClient(server_params)

tools = await client.get_tools()

app = langgraph_agent.buildChatAgentWithHistory(tools)
config = {
    "recursion_limit": 50,
    "configurable": {"thread_id": user_id},
    "tools": tools,
    "system_prompt": None
}

inputs = {
    "messages": [HumanMessage(content=query)]
}

result = ""
async for output in app.astream(inputs, config, stream_mode="messages"):
  message = output[0]
  for content_item in message.content:
    if content_item.get('type') == 'text':
      text_content = content_item.get('text', '')
      result += text_content
```



## 실행 하기

1) 소스를 다운로드 합니다.

```text
git clone https://github.com/kyopark2014/lgm-project
```

2) config.json을 생성합니다.

```text
cd lgm-project && cp application/config.json.sample application/config.json
```

3) [knowledge_base.md](https://github.com/kyopark2014/lgm-project/blob/main/knowledge_base.md)에 따라 Knowledge Base를 설치합니다. 설치가 완료되면 아래와 같이 Knowledge Base ID를 확인할 수 있습니다.


<img width="687" height="276" alt="noname" src="https://github.com/user-attachments/assets/31112632-b4fd-45f6-b6ae-d74e889e9667" />


4) AWS CLI를 통해 아래의 Knowledge Based ID와 AWS Credential을 설정합니다. 만약, AWS CLI를 사용하기 어려운 환경이라면(예 워크샵 계정) config.json 파일을 열어서 access_key_id, secret_access_key, session_token을 입력합니다.

```java
{
    "projectName":"mcp",
    "knowledge_base_id":"1CMBJP5NME",
    "region":"us-west-2"
 }
```

이제 아래와 같이 streamlit으로 된 application을 실행할 수 있습니다. 

```text
streamlit run application/app.py
```



### 실행 예제

왼쪽 메뉴에서 Agent와 "Knowledge Base" MCP를 선택하고 질문을 입력하면, RAG로 부터 문서를 조회하여 아래와 같은 결과를 얻을 수 있습니다.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/0c8064cc-00ff-4d80-ac6c-01ff3b4c1d97" />

MCP로 "code interpreter"를 선택한 후에 "DNA의 나선형 구조를 그려주세요."을 입력합니다. 적절한 code를 생성하여 repl_coder를 이용해 아래와 같은 그림을 그릴 수 있습니다.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/cb3988b3-3f03-4e75-993b-d380d2ef3ad7" />

왼쪽 메뉴에서 "QA Agent"를 선택하고 아래와 같이 질문을 하면 RAG에서 얻어진 문서를 바탕으로 Test case들을 생성할 수 있습니다.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/c5f0deb5-8f56-458e-a5d5-f0b953e90c6f" />


