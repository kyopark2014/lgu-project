import logging
import sys
import json
import boto3
import os
import time
import s3vector

from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("utils")

aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
aws_session_token = os.environ.get('AWS_SESSION_TOKEN')

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.json")

def load_config():
    config = None
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        config = {}
        config['projectName'] = "lgu"
        config['region'] = "us-west-2"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    
    return config

config = load_config()

accountId = config.get('accountId')
# Strengthen region resolution priority: boto3 session -> env var -> existing config -> default
_env_region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
_session = boto3.Session()
_session_region = _session.region_name
resolved_region = _session_region or _env_region or config.get('region') or "us-west-2"

if not accountId:
    sts = boto3.client("sts")
    response = sts.get_caller_identity()
    accountId = response["Account"]
    config['accountId'] = accountId
    config['region'] = resolved_region
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
else:
    # If accountId exists but region is empty/None, restore with resolved value
    if not config.get('region'):
        config['region'] = resolved_region
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

bedrock_region = config.get("region") or _env_region or "us-west-2"
# If region in config is missing/None, persist the corrected value
if config.get("region") != bedrock_region:
    config['region'] = bedrock_region
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
logger.info(f"bedrock_region: {bedrock_region}")
projectName = config.get("projectName", "")
logger.info(f"projectName: {projectName}")

# Bucket for Knowledge Base
bucket_name = config.get("bucket_name", "")
logger.info(f"bucket_name: {bucket_name}")

if not bucket_name:
    bucket_name = f"storage-for-{projectName}-{accountId}-{bedrock_region}"
    config['bucket_name'] = bucket_name
    logger.info(f"bucket_name: {bucket_name}")
    
    s3vector.create_bucket(bucket_name, bedrock_region)
    
    # write bucket name
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

# Knowledge Base
knowledge_base_id = config.get('knowledge_base_id', "")
logger.info(f"knowledge_base_id: {knowledge_base_id}")    
if not knowledge_base_id:
    logger.info(f"knowledge_base_id is required.")    
    knowledge_base_name = projectName
    logger.info(f"knowledge_base_name: {knowledge_base_name}")
    s3vector.create_knowledge_base(knowledge_base_name, bedrock_region)
        
def get_contents_type(file_name):
    if file_name.lower().endswith((".jpg", ".jpeg")):
        content_type = "image/jpeg"
    elif file_name.lower().endswith((".pdf")):
        content_type = "application/pdf"
    elif file_name.lower().endswith((".txt")):
        content_type = "text/plain"
    elif file_name.lower().endswith((".csv")):
        content_type = "text/csv"
    elif file_name.lower().endswith((".ppt", ".pptx")):
        content_type = "application/vnd.ms-powerpoint"
    elif file_name.lower().endswith((".doc", ".docx")):
        content_type = "application/msword"
    elif file_name.lower().endswith((".xls")):
        content_type = "application/vnd.ms-excel"
    elif file_name.lower().endswith((".py")):
        content_type = "text/x-python"
    elif file_name.lower().endswith((".js")):
        content_type = "application/javascript"
    elif file_name.lower().endswith((".md")):
        content_type = "text/markdown"
    elif file_name.lower().endswith((".png")):
        content_type = "image/png"
    else:
        content_type = "no info"    
    return content_type

# api key to get weather information in agent
if aws_access_key and aws_secret_key:
    secretsmanager = boto3.client(
        service_name='secretsmanager',
        region_name=bedrock_region,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        aws_session_token=aws_session_token,
    )
else:
    secretsmanager = boto3.client(
        service_name='secretsmanager',
        region_name=bedrock_region
    )

# api key for weather
weather_api_key = ""
try:
    get_weather_api_secret = secretsmanager.get_secret_value(
        SecretId=f"openweathermap-{projectName}"
    )
    #print('get_weather_api_secret: ', get_weather_api_secret)
    secret = json.loads(get_weather_api_secret['SecretString'])
    #print('secret: ', secret)
    weather_api_key = secret['weather_api_key']

except Exception as e:
    # raise e
    pass

# api key to use Tavily Search
tavily_key = tavily_api_wrapper = ""
try:
    get_tavily_api_secret = secretsmanager.get_secret_value(
        SecretId=f"tavilyapikey-{projectName}"
    )
    #print('get_tavily_api_secret: ', get_tavily_api_secret)
    secret = json.loads(get_tavily_api_secret['SecretString'])
    #print('secret: ', secret)

    if "tavily_api_key" in secret:
        tavily_key = secret['tavily_api_key']
        #print('tavily_api_key: ', tavily_api_key)

        if tavily_key:
            tavily_api_wrapper = TavilySearchAPIWrapper(tavily_api_key=tavily_key)
            #     os.environ["TAVILY_API_KEY"] = tavily_key

        else:
            logger.info(f"tavily_key is required.")
except Exception as e: 
    logger.info(f"Tavily credential is required: {e}")
    # raise e
    pass

