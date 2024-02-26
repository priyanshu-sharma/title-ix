import json
import ollama
import pandas as pd

df = pd.read_csv('../dataset_domain/data.csv')
states = df['state'].unique()

def get_help_metrics():
    return "{}\nHere is the response for the above scenario - {}\nCan you rate this information based on the help provided by the generated response on a scale of 0-10, where 0 is for lowest level of information, while 10 for the most information. Only provide the score nothing else. No extra information or explaination."

def get_humanlike_metrics():
    return "{}\nHere is the response for the above scenario - {}\nCan you evaluate how closely the response resembles human-generated conversation on a scale of 0-10, where 0 is for lowest level of human resemblances, while 10 for the most resemblances. Only provide the score nothing else. No extra information or explaination."

def get_relevance_metrics():
    return "{}\nHere is the response for the above scenario - {}\nCan you evaluate how relevant the response is to the above scenario. Rate on a scale of 0-10, where 0 is for lowest level of relevance, while 10 for the most relevance. Only provide the score nothing else. No extra information or explaination."

def get_contextual_metrics():
    return "{}\nHere is the response for the above scenario - {}\nCan you evaluate how well the contextual information of the above scenario is being utilized in the above response. Rate on a scale of 0-10, where 0 is for lowest level of contextual information, while 10 for the most. Only provide the score nothing else. No extra information or explaination."

def get_empathy_metrics():
    return "{}\nHere is the response for the above scenario - {}\nCan you evaluate how well this response empathetically to the victims's emotions and concerns. Rate on a scale of 0-10, where 0 is for lowest level of empathy, while 10 for the most. Only provide the score nothing else. No extra information or explaination."

for state in states:
    with open('../datadump/{}.json'.format(state), 'r') as openfile:
        json_object = json.load(openfile)
    response_list = []
    for data in json_object:
        question = data['Question']
        answer = data['Response']
        help_content = get_help_metrics()
        help_content = help_content.format(question, answer)
        help_content_response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': help_content}])
        help_content_result = help_content_response['message']['content']
        human_content = get_humanlike_metrics()
        human_content = human_content.format(question, answer)
        human_content_response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': human_content}])
        human_content_result = human_content_response['message']['content']
        relevance_content = get_relevance_metrics()
        relevance_content = relevance_content.format(question, answer)
        relevance_content_response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': relevance_content}])
        relevance_content_result = relevance_content_response['message']['content']
        contextual_content, contextual_contents_metrices = get_contextual_metrics()
        contextual_content = contextual_content.format(question, answer)
        contextual_content_response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': contextual_content}])
        contextual_content_result = contextual_content_response['message']['content']
        empathy_content = get_empathy_metrics()
        empathy_content = empathy_content.format(question, answer)
        empathy_content_response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': empathy_content}])
        empathy_content_result = empathy_content_response['message']['content']
        response_list.append({'Question': question, 'Response': answer, 'helpful_score': help_content_result, 'humanlike_score': human_content_result, 'relevance_score': relevance_content_result, 'contextual_score': contextual_content_result, 'empathy_score': empathy_content_result})
    with open("../datadump/{}_new_scoring.json".format(state), "w") as f:
        json.dump(response_list, f)
    with open("../datadump/{}_new_scoring.json".format(state), 'r') as openfile:
        json_object = json.load(openfile)
    final = ''
    for values in json_object:
        final = final + '\nQuestion : - {}\n\nAnswer : - {}\n\nHelpfulness : - {}\n\nHuman-Likeness : - {}\n\nRelevance : - {}\n\nContextual Information : - {}\n\nEmpathy : - {}\n'.format(values['Question'], values['Response'], values['helpful_score'], values['humanlike_score'], values['relevance_score'], values['contextual_score'], values['empathy_score'])
    with open("../datadump/{}_new_scoring.txt".format(state), "w", encoding="utf-8") as f:
        f.write(final)