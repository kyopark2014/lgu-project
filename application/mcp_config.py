import logging
import sys
import os

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("mcp-config")

workingDir = os.path.dirname(os.path.abspath(__file__))
logger.info(f"workingDir: {workingDir}")

mcp_user_config = {}    

def load_config(mcp_type):
    if mcp_type == "code interpreter":
        mcp_type = "repl_coder"
    elif mcp_type == "RAG":
        mcp_type = "kb-retriever"
    elif mcp_type == "use-aws":
        mcp_type = "use-aws"
    
    if mcp_type == "kb-retriever":
        return {
            "mcpServers": {
                "kb_retriever": {
                    "command": "python",
                    "args": [f"{workingDir}/mcp_server_retrieve.py"]
                }
            }
        }
    
    elif mcp_type == "repl_coder":
        return {
            "mcpServers": {
                "repl_coder": {
                    "command": "python",
                    "args": [
                        f"{workingDir}/mcp_server_repl_coder.py"
                    ]
                }
            }
        }    

    elif mcp_type == "terminal":
        return {
            "mcpServers": {
                "iterm-mcp": {
                    "command": "npx",
                    "args": [
                        "-y",
                        "iterm-mcp"
                    ]
                }
            }
        }
    
    elif mcp_type == "filesystem":
        return {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": [
                        "@modelcontextprotocol/server-filesystem",
                        f"{workingDir}"
                    ]
                }
            }
        }
    
    elif mcp_type == "aws_documentation":
        return {
            "mcpServers": {
                "awslabs.aws-documentation-mcp-server": {
                    "command": "uvx",
                    "args": ["awslabs.aws-documentation-mcp-server@latest"],
                    "env": {
                        "FASTMCP_LOG_LEVEL": "ERROR"
                    }
                }
            }
        }
    
    elif mcp_type == "use-aws":
        return {
            "mcpServers": {
                "use-aws": {
                    "command": "python",
                    "args": [
                        f"{workingDir}/mcp_server_use_aws.py"
                    ]
                }
            }
        }    
    
    elif mcp_type == "사용자 설정":
        return mcp_user_config

def load_selected_config(mcp_servers: dict):
    logger.info(f"mcp_servers: {mcp_servers}")
    
    loaded_config = {}
    for server in mcp_servers:
        config = load_config(server)
        if config:
            loaded_config.update(config["mcpServers"])
    return {
        "mcpServers": loaded_config
    }
