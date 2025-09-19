import logging
import sys
import json
import boto3
import os
import time

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
if not accountId:
    session = boto3.Session()
    region = session.region_name

    sts = boto3.client("sts")
    response = sts.get_caller_identity()
    accountId = response["Account"]
    config['accountId'] = accountId
    config['region'] = region
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

bedrock_region = config.get("region", "")
logger.info(f"bedrock_region: {bedrock_region}")
projectName = config.get("projectName", "")
logger.info(f"projectName: {projectName}")

# Bucket for Knowledge Base
bucket_name = config.get("bucket_name", "")
logger.info(f"bucket_name: {bucket_name}")

if not bucket_name:
    bucket_name = f"storage-for-{projectName}-{accountId}-{bedrock_region}"
    config['bucket_name'] = bucket_name
    
    s3 = boto3.client('s3', region_name=bedrock_region)
    response = s3.list_buckets()
    buckets = response.get('Buckets', [])
    if not any(bucket['Name'] == bucket_name for bucket in buckets):
        logger.info(f"bucket_name: {bucket_name} is not exists.")
        
        s3.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={'LocationConstraint': bedrock_region}
        )
        logger.info(f"bucket_name: {bucket_name} is created.")    
        
        s3.put_object(Bucket=bucket_name, Key='docs/')
        logger.info(f"docs folder is created in {bucket_name}.")
        
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

# Knowledge Base
knowledge_base_id = config.get('knowledge_base_id')
logger.info(f"knowledge_base_id: {knowledge_base_id}")    

if not knowledge_base_id:
    logger.info(f"knowledge_base_id is required.")
    
    knowledge_base_name = projectName
    config['knowledge_base_name'] = knowledge_base_name
    
    role_name = f"role-knowledge-base-for-{projectName}-{bedrock_region}"
    iam = boto3.client('iam', region_name=bedrock_region)
    response = iam.list_roles(MaxItems=200)
    # logger.info(f"response of list_roles: {response}")
    
    roles = response.get('Roles', [])
    if not any(role['RoleName'] == role_name for role in roles):
        logger.info(f"role_name: {role_name} is not exists.")
        response = iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "bedrock.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }
            ]
        }))
        role_arn = response['Role']['Arn']
        config['knowledge_base_role_arn'] = role_arn
        
        iam.put_role_policy(RoleName=role_name, PolicyName=f"knowledge-base-for-{projectName}-{bedrock_region}", PolicyDocument=json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "BedrockAllStatement",
                    "Action": "bedrock:*",
                    "Resource": "*",
                    "Effect": "Allow"
                },
                {
                    "Sid": "BedrockInvokeInferenceProfileStatement",
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:GetInferenceProfile",
                        "bedrock:InvokeModel"
                    ],
                    "Resource": "*"
                },
                {
                    "Sid": "S3BucketStatement",
                    "Effect": "Allow",
                    "Action": ["s3:*"],
                    "Resource": "*"
                }
            ]
        }))
        
        logger.info(f"role_name: {role_name} is created.")
    
    client = boto3.client('s3vectors')
    
    s3_vector_bucket_name = config.get('s3_vector_bucket_name', "")
    s3_vector_bucket_arn = config.get('s3_vector_bucket_arn', "")
    if not s3_vector_bucket_name or not s3_vector_bucket_arn:
        s3_vector_bucket_name = f"s3-vector-for-{projectName}-{accountId}-{bedrock_region}"
        logger.info(f"s3_vector_bucket_name: {s3_vector_bucket_name}")
        config['s3_vector_bucket_name'] = s3_vector_bucket_name
                
        response = client.list_vector_buckets(maxResults=50)
        logger.info(f"response of list_vector_buckets: {response}")
        
        vectorBuckets = response.get('vectorBuckets', [])
        if any(vectorBucket['vectorBucketName'] == s3_vector_bucket_name for vectorBucket in vectorBuckets):
            for vectorBucket in vectorBuckets:
                if vectorBucket['vectorBucketName'] == s3_vector_bucket_name:
                    s3_vector_bucket_arn = vectorBucket['vectorBucketArn']
                    logger.info(f"s3_vector_bucket_arn: {s3_vector_bucket_arn}")
                    break
        else:
            response = client.create_vector_bucket(vectorBucketName=s3_vector_bucket_name)
            logger.info(f"response of create_vector_bucket: {response}")
            logger.info(f"s3_vector_bucket_name: {s3_vector_bucket_name} is created.")
            s3_vector_bucket_arn = response['vectorBucketArn']
            logger.info(f"s3_vector_bucket_arn: {s3_vector_bucket_arn}")
            
        config['s3_vector_bucket_arn'] = s3_vector_bucket_arn
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            
    s3_vector_index_name = config.get('s3_vector_index_name', "")    
    s3_vector_index_arn = config.get('s3_vector_index_arn', "")
    
    if not s3_vector_index_name:
        s3_vector_index_name = f"s3-vector-index-for-{projectName}-{accountId}-{bedrock_region}"
        logger.info(f"s3_vector_index_name: {s3_vector_index_name}")
        config['s3_vector_index_name'] = s3_vector_index_name
        
        response = client.list_indexes(vectorBucketName=s3_vector_bucket_name)
        indexes = response.get('indexes', [])
        
        if any(index['indexName'] == s3_vector_index_name for index in indexes):
            for index in indexes:
                if index['indexName'] == s3_vector_index_name:
                    s3_vector_index_arn = index['indexArn']
                    logger.info(f"s3_vector_index_arn: {s3_vector_index_arn}")
                    break
        else:
            response = client.create_index(
                vectorBucketArn=s3_vector_bucket_arn,
                indexName=s3_vector_index_name,
                dataType='float32',
                dimension=1024,
                distanceMetric='cosine'                
            )
            logger.info(f"response of create_index: {response}")
            time.sleep(3)
            
            response = client.list_indexes(vectorBucketName=s3_vector_bucket_name)
            indexes = response.get('indexes', [])
            for index in indexes:
                if index['indexName'] == s3_vector_index_name:
                    s3_vector_index_arn = index['indexArn']
                    logger.info(f"s3_vector_index_arn: {s3_vector_index_arn}")
                    break
            
        config['s3_vector_index_arn'] = s3_vector_index_arn
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    elif not s3_vector_index_arn:
        # 인덱스 이름은 있지만 ARN이 없는 경우 ARN을 가져옴
        response = client.list_indexes(vectorBucketName=s3_vector_bucket_name)
        indexes = response.get('indexes', [])
        for index in indexes:
            if index['indexName'] == s3_vector_index_name:
                s3_vector_index_arn = index['indexArn']
                logger.info(f"s3_vector_index_arn: {s3_vector_index_arn}")
                config['s3_vector_index_arn'] = s3_vector_index_arn
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2)
                break
            
    knowledge_base_id = boto3.client('bedrock-agent').create_knowledge_base(
        name=f"knowledge-base-for-{projectName}-{bedrock_region}",
        description=f"Knowledge base for {knowledge_base_name}",
        roleArn=f"arn:aws:iam::{accountId}:role/{role_name}",
        knowledgeBaseConfiguration={
            "type": "VECTOR",
            "vectorKnowledgeBaseConfiguration": {
                "embeddingModelArn": f"arn:aws:bedrock:{bedrock_region}:{accountId}:model/amazon.titan-embed-text-v2:0", 
                "embeddingModelConfiguration": {
                    "bedrockEmbeddingModelConfiguration": {
                        "dimensions": 1024,
                        "embeddingDataType": "FLOAT32"
                    }
                }
            }
        },
        storageConfiguration={
            "type": "S3_VECTORS",
            "s3VectorsConfiguration": {
                "vectorBucketArn": s3_vector_bucket_arn,
                "indexArn": s3_vector_index_arn,
                "indexName": s3_vector_index_name
            }
        }
    )
    config['knowledge_base_id'] = knowledge_base_id
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

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

def load_mcp_env():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mcp_env_path = os.path.join(script_dir, "mcp.env")
    
    with open(mcp_env_path, "r", encoding="utf-8") as f:
        mcp_env = json.load(f)
    return mcp_env

def save_mcp_env(mcp_env):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mcp_env_path = os.path.join(script_dir, "mcp.env")
    
    with open(mcp_env_path, "w", encoding="utf-8") as f:
        json.dump(mcp_env, f)

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

