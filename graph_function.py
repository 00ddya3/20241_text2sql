# backlogs
# retriever 진행 후 찾은 top k와 max 유사도가 일정 이상이라면 few shot prompt, 미만이라면 적합한 db schema 재검색 
# CoT를 단계를 명확하게 포맷팅해서 진행시키도록 프롬프트 수정
# 스트림릿 서빙 시 ModuleNotFoundError: No module named 'distutils'

# errorlogs
# router prompt의 분류 오류가 잦음
# 오늘꺼 검색해줘 -> db에 적힌 날짜를 그대로 반영하는 문제

import os
import json
import boto3
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

import opensearchpy
from opensearchpy import OpenSearch
from opensearchpy.helpers import scan
from openai import OpenAI


dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_SESSION_TOKEN'] = os.getenv('AWS_SESSION_TOKEN')
os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_AI_KEY')

hosts = [{'host': os.getenv('AWS_opensearch_Domain_Endpoint'), 'port': 443}]
print(hosts)
opensearch_client = OpenSearch(
    hosts=hosts,
    http_auth=(os.getenv('AWS_opensearch_ID'), os.getenv('AWS_opensearch_PassWord')),
    use_ssl=True,
    verify_certs=True,
    ssl_show_warn=False,
    timeout=30 #30초 이상 서치하면 넘나 길다.
)

chat_history = []

def LLM(LLM_input):
    global chat_history  # 전역 채팅 기록 사용

    # 채팅 기록에 새 메시지 추가
    chat_history.append({"role": "user", "content": LLM_input})

    # AWS Bedrock 클라이언트 생성
    client = boto3.client('bedrock-runtime', region_name='us-east-1')
    bedrock_client = boto3.client('bedrock', region_name='us-east-1')

    # 요청 본문 작성
    request_body = {
        "max_tokens": 8192,
        "messages": chat_history,  # 채팅 기록 포함
        "anthropic_version": "bedrock-2023-05-31"
    }

    # 모델 호출
    response = client.invoke_model(
        modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',  # 사용할 모델 ID
        body=json.dumps(request_body)
    )

    # 응답 처리
    response_body = response['body'].read()
    response_json = json.loads(response_body)
    output = response_json['content'][0]['text']

    # 응답을 채팅 기록에 추가
    chat_history.append({"role": "assistant", "content": output})

    print(output)  # 응답 출력
    return output


def LLM_get_embedding(text, model_name="text-embedding-3-large"):
    client = OpenAI(api_key=os.getenv('OPEN_AI_KEY'))
    response = client.embeddings.create(input=text,model=model_name)
    print(text)
    return response.data[0].embedding


def LLM_Router(state):
    print(f"LLM_Router가 질문 분류 중..")
    user_question = state["user_question"]
    #history.add_user_message(user_question)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'prompt_files/router.txt')

    with open(file_path, 'r', encoding='utf-8') as file:
        llm_input = file.read()
    #llm_input = llm_input.replace('{chat_history}', str(history.messages))
    llm_input = llm_input.replace('{user_question}', user_question)  

    user_intent = LLM(llm_input)
    state["user_intent"] = user_intent
    print(f"LLM_Router가 {user_intent}로 가라고 합니다")
    return state

"""
def LLM_common(state):
    user_question = state["user_question"]
    llm_output = LLM(user_question)
    state['final_output'] = llm_output
    return state
"""

def LLM_event_list(state):
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'event_crawling.csv')
    events_crawled = pd.read_csv(file_path)
    user_question = state["user_question"]

    # prompt 작성
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'prompt_files/event_list_generation.txt')
    with open(file_path, 'r', encoding='utf-8') as file:
        llm_input = file.read()
    llm_input = llm_input.replace('{current_time}', current_time)    
    #llm_input = llm_input.replace('{events_crawled}', str(events_crawled))#리스트여서 문자열로 바꿔 줌
    llm_input = llm_input.replace('{user_question}', user_question)    

    events_condition = LLM(llm_input)
    print(events_condition)
    exec(str(events_condition))

    events_output = events_crawled[['title', 'k_date', 'place']]
    print(events_output.shape)
    
    events_output_str = ''
    for index, row in events_output.iterrows():
        events_output_str = events_output_str + f"{index}. 제목: {row['title']} 날짜: {row['k_date']} 위치: {row['place']}  " + '\n'

    state["events_output"] = events_output_str
    #history.add_ai_message(events_output_str)
    print(f"LLM_event_list가 뽑은 이벤트 목록 : {events_output_str}")

    return state


def Retrieve(state):
    print(f"Retrieve 가 검색하는 중")

    KDB_index = 'kdbtest_vectorized_tokenized_jihoon'
    print(KDB_index)
    # 이 KDB index에서부터 qa데이터를 뽑아옴. 요거는 지훈이 만든 토크나이징 룰 + 사전 패키지 기반으로 만들어진 인덱스임. 
    # 사전 목록을 바꾸고 싶으면, jihoon-dictionary패키지를 업데이트 해야 함.
    # jihoon-dictionary패키지는, s3://infra-ai-assistant-opensearch/jihoon_dictionary.txt를 참조하고 있음.
    # s3://infra-ai-assistant-opensearch/jihoon_dictionary.txt를 업데이트 한 다음 패키지를 업데이트하면 업데이트된 새로운 룰로 토크나이징함.

    mapping = opensearch_client.indices.get_mapping(index=KDB_index)
    print(f"검색하려는 KDB INDEX 이름 : {KDB_index}")
    #print(f"KDB INDEX 구조 : {json.dumps(mapping, indent=2)}")

    response = opensearch_client.indices.get(index=KDB_index)
    settings = response[KDB_index]['settings']['index']['analysis']
    analyzer_setting = settings['analyzer']
    analyzer_name = str(list(analyzer_setting.keys())[0])
    tokenizer_setting = settings['tokenizer']
    tokenizer_name = str(list(tokenizer_setting.keys())[0])
    print(f"인덱스 이름 : {KDB_index}")
    print(f"이 인덱스의 lexical 세팅값\n")
    print("<<<<<<<<<<<<<<<<<<<<<<<")
    print(analyzer_setting)
    print(tokenizer_setting)
    print(">>>>>>>>>>>>>>>>>>>>>>>\n")


    def lexical_analyze(index_name, text, analyzer_name): 
        #토크나이징 결과를 시각적으로 확인하기 위한 분석 쿼리. 실제로는 lexical_search도중에 이미 토크나이징이 되지만, 
        #토크나이징된 리스트 확인이 lexical_search의 response로 확인이 안 돼서 만듦
        analyze_query = {
            "analyzer": analyzer_name,
            "text": text
        }
        response = opensearch_client.indices.analyze(index=index_name, body=analyze_query)
        return [token['token'] for token in response['tokens']]
    
    def lexical_search(index_name, user_query, size=3):
        search_query = {
            'query': {
                'match': {
                    'question': user_query  # Assuming the documents have a field named 'content'
                }
            },
            'size': size
        }
        response = opensearch_client.search(index=index_name, body=search_query)
        return response['hits']['hits']
    
    def vector_search(index_name, user_query, size=3):
        query_vector = LLM_get_embedding(user_query, model_name="text-embedding-3-large")
        search_query = {
            "size": size,
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "knn_score",
                        "lang": "knn",
                        "params": {
                            "field": "question_vector",
                            "query_value": query_vector,
                            "space_type": "cosinesimil"
                        }
                    }
                }
            }
        }
        response = opensearch_client.search(index=index_name, body=search_query)
        return response['hits']['hits']
    
    user_question = state["user_question"]
    lexical_searched_data = lexical_search(KDB_index, user_question, size=3)
    vector_searched_data = vector_search(KDB_index, user_question, size=3)

    lexical_search_result, vector_search_result = [], []

    for data in lexical_searched_data:
        lexical_search_result.append((data['_source']['question'], data['_source']['query']))
    for data in vector_searched_data:
        vector_search_result.append((data['_source']['question'], data['_source']['query']))
    
    #print(f"Lexical Retrieve 가 검색한 데이터 k개 : {lexical_search_result}")
    #print(f"Vector Retrieve 가 검색한 데이터 k개 : {vector_search_result}")

    state["top_k"] = lexical_search_result + vector_search_result 
    
    top_k_str = ''
    for k, (DB_question, DB_query) in enumerate(state['top_k'], start=1):
        top_k_str = top_k_str + f"question: {DB_question} \t query: {DB_query}"
    #history.add_ai_message(top_k_str)

    user_question_tokens = lexical_analyze(KDB_index, user_question, analyzer_name)
    retrieved_question_tokens = [lexical_analyze(KDB_index, retrieved_question[0], analyzer_name) for retrieved_question in state["top_k"]]

    """
    print('-'*100)
    print(f"유저 입력 자연어 : {user_question}")
    print(f"유저 입력 자연어의 토큰화 결과 : {user_question_tokens}")
    print('-'*100)

    print("검색된 자연어 :")
    for question, query in state["top_k"]:
        print(question)

    print('-'*100)
    for retrieved_question_token in retrieved_question_tokens:
        print(f"검색된 자연어의 토큰화 결과 : {retrieved_question_token}")
    """
    return state 


def LLM_Final_Generate(state):
    print(f"LLM_Final_Generate가 최종 생성하려는 중")
    user_question = state["user_question"]
    retrieved_top_k = state["top_k"] # retrieve된 top_k_rows
 
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'prompt_files/final_generation.txt')
    with open(file_path, 'r', encoding='utf-8') as file:
        llm_input = file.read()

    llm_input = llm_input.replace('{user_question}', user_question)
    #llm_input = llm_input.replace('{chat_history}', str(MessagesPlaceholder("chat_history")))
    llm_input = llm_input.replace('{retrieved_top_k}', str(retrieved_top_k))    #리스트여서 문자열로 바꿔 줌


    final_output = LLM(llm_input) # (instruction + retrieve된 top_k_rows + 실제 유저 입력)을 통해 최종 sql문과 CoT를 통한 생성원인을 출력
    print(f"LLM_Final_Generate가 최종 생성함 : {final_output}")
    state["final_output"] = final_output
    
    # SQL 코드만 chat history에 저장
    sql_output = final_output.split('```sql')[1].split('```')[0].strip()
    #history.add_ai_message(sql_output)

    #print(history.messages)

    return state