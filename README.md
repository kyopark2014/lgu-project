# Agent and MCP

여기에서는 [Strands SDK](https://strandsagents.com/latest/documentation/docs/) 기반의 Agent와 MCP를 활용하는 방법에 대해 설명합니다. 전체적인 architecture는 아래와 같습니다. 개발의 편의를 위하여 [Streamlit](https://streamlit.io/)을 이용해 UI를 구성하고, stdio 방식의 custom MCP 서버를 정의하여 활용합니다. 여기에서는 강력하면서도 쉽게 사용할 수 있는 완전관리형 RAG 서비스인 Knowledge Base를 [kb-retriever](./application/mcp_server_retrieve.py)를 이용해 RAG를 활용합니다. 복잡한 데이터의 분석을 수행하는 Code Interpreter는 [repl-coder](./application/mcp_server_repl_coder.py)을 이용해 구현하였고, AWS의 각종 문서를 MCP로 조회하는 [AWS Documentation](https://awslabs.github.io/mcp/servers/aws-documentation-mcp-server/)을 지원합니다.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/70c4d8b0-8ff3-4c9f-aa4f-a4719e3e5a8b" />

## MCP Agent 구현하기

아래에서는 MCP와 Strands SDK를 이용해 RAG 등을 활용할 수 있는 Agent를 구현하는 방법에 대해 설명합니다.

### Agent

Strands SDK에 기반한 agent는 아래와 같이 model을 먼저 설정합니다. 상세한 코드는 [strands_agent.py](./application/strands_agent.py)을 참조합니다.

```python
def get_model():
    STOP_SEQUENCE = "\n\nHuman:" 
    maxOutputTokens = 4096 

    bedrock_config = Config(
        read_timeout=900,
        connect_timeout=900,
        retries=dict(max_attempts=3, mode="adaptive"),
    )

    bedrock_client = boto3.client(
        'bedrock-runtime',
        region_name=aws_region,
        config=bedrock_config
    )

    model = BedrockModel(
        client=bedrock_client,
        model_id=chat.model_id,
        max_tokens=maxOutputTokens,
        stop_sequences = [STOP_SEQUENCE],
        temperature = 0.1,
        top_p = 0.9,
        additional_request_fields={
            "thinking": {
                "type": "disabled"
            }
        }
    )
    return model
```

이후 아래와 같이 agent를 생성합니다. Agent믄 model, system prompt, tools 정보를 설정하고, 이전 히스토리를 참조할 때에는 conversation manager를 아래와 같이 설정합니다. 

```python
from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager

conversation_manager = SlidingWindowConversationManager(
    window_size=10,  
)

def create_agent(system_prompt, tools, history_mode):
    if system_prompt==None:
        system_prompt = (
            "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
            "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
            "모르는 질문을 받으면 솔직히 모른다고 말합니다."
        )

    model = get_model()
    if history_mode == "Enable":
        agent = Agent(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            conversation_manager=conversation_manager
        )
    else:
        agent = Agent(
            model=model,
            system_prompt=system_prompt,
            tools=tools
        )
    return agent
```

이후 아래와 같이 stream_async로 실행한 후에 stream 결과를 활용합니다.

```python
agent_stream = strands_agent.agent.stream_async(query)
        
async for event in agent_stream:
    result = text = ""            
    if "data" in event:
        text = event["data"]
        result += text
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



## 실행 하기

1) 소스를 다운로드 합니다.

```text
git clone https://github.com/kyopark2014/lgu-project && cd lgu-project
```

이제 아래와 같이 streamlit으로 된 application을 실행할 수 있습니다. 

```text
streamlit run application/app.py
```



### 실행 예제

왼쪽 메뉴에서 Agent와 "RAG" MCP를 선택하고 질문을 입력하면, RAG로 부터 문서를 조회하여 아래와 같은 결과를 얻을 수 있습니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/0c8064cc-00ff-4d80-ac6c-01ff3b4c1d97" />

MCP로 "code interpreter"를 선택한 후에 "DNA의 나선형 구조를 그려주세요."을 입력합니다. 적절한 code를 생성하여 repl_coder를 이용해 아래와 같은 그림을 그릴 수 있습니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/cb3988b3-3f03-4e75-993b-d380d2ef3ad7" />

