# backlog
# chat history limit 설정

import streamlit as st
import sqlparse
from langchain.memory import StreamlitChatMessageHistory
import graph_function
import graph_structure


st.set_page_config(page_title="text2sql webpage")
st.title("Text2SQL Chatbot")

# credential 정보 입력
    

with st.sidebar :
    st.subheader("graph structure")
    st.image(graph_structure.my_graph_image(graph_structure.my_graph()))
    st.write("intent detection을 통해 사용자의 입력이 sql 분석을 요청한다면 Lexical+Vector서치 후 sql문 생성, \
              이벤트 리스트를 요청한다면 event DB 검색을 통한 이벤트 리스트 반환, \
              앞선 대화에 관한 내용이라면 chat history를 참조해서 적합한 답변을 생성한다.")

st.session_state['conversation'] = None
st.session_state['chat_history'] = None

# assistant의 시작 메세지
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", 
                                    "content": "외계인 침공 시 SQL 못 하는 사람이 먼저 잡아 먹힌다 :alien: (제발 저요)\n\
                                        예제: 제주 Access Infra팀이 관리하고 있는 LTE 기지국의 ID와 ENB ID를 말해줘"}]
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#history = StreamlitChatMessageHistory(key="chat_messages")

# Chat logic
if user_question := st.chat_input("무엇이든 물어보살"):
    st.session_state.messages.append({"role": "user", "content": user_question})
    #history.add_user_message(user_question)

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        chain = st.session_state.conversation

        inputs = graph_structure.GraphState(user_question = user_question)


        for output in graph_structure.my_graph().stream(inputs):
            for node, state in output.items():
                if node == "LLM_Router":
                    st.markdown(f":alien: 쉿! '{node}' 진행 중. :alien:")
                    st.markdown(f"user_intent: {state['user_intent']}")
                    st.markdown("\n\n")
                    
                elif node == "LLM_event_list":
                    st.markdown(f":alien: 쉿! '{node}' 진행 중. :alien:")
                    st.markdown(f"events_output: {state['events_output']}")
                    #history.add_ai_message(state['events_output'])
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.markdown("\n\n")

                elif node == 'Retrieve':
                    st.markdown(f":alien: 쉿! '{node}' 진행 중. :alien:")
                    st.markdown('검색된 관련 질문')

                    top_k_str=''

                    for k, (DB_question, DB_query) in enumerate(state['top_k'], start=1):
                        st.markdown(f"{k}:  {DB_question}")
                        top_k_str = top_k_str + f"question: {DB_question} \t query: {DB_query}"
                        #st.markdown(f"검색된 관련 질문 {k}: {DB_question}")
                        st.markdown("\n")

                    #history.add_ai_message(top_k_str)
                    st.markdown("\n\n")
                    

                elif node == 'LLM_Final_Generate':
                    st.markdown(f":alien: 쉿! '{node}' 진행 중. :alien:")
                    st.markdown("\n\n")
                    text = state["final_output"]
                    
                    # SQL 코드 포맷팅
                    raw_sql = text.split('```sql')[1].split('```')[0].strip()
                    formatted_sql = sqlparse.format(raw_sql, reindent=True, keyword_case='upper')
                    response = text.replace(raw_sql, formatted_sql)

                    # Streamlit에 표시
                    st.markdown(response, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    #history.add_ai_message(formatted_sql)

#print('\n\n\nhistory: \n', history.messages)
                
