import json
import ollama
import pandas as pd

df = pd.read_csv('../dataset_domain/data.csv')
states = df['state'].unique()

def get_help_metrics():
    return "{}\nHere is the response for the above scenario - {}\nCan you rate this information based on the help provided by the generated response on a scale of 0-10, where 0 is for lowest level of information, while 10 for the most information. Only provide the score nothing else. No extra information or explaination.", "helpful_score"

def get_humanlike_metrics():
    return "{}\nHere is the response for the above scenario - {}\nCan you evaluate how closely the response resembles human-generated conversation on a scale of 0-10, where 0 is for lowest level of human resemblances, while 10 for the most resemblances. Only provide the score nothing else. No extra information or explaination.", "humanlike_score"

def get_relevance_metrics():
    return "{}\nHere is the response for the above scenario - {}\nCan you evaluate how relevant the response is to the above scenario. Rate on a scale of 0-10, where 0 is for lowest level of relevance, while 10 for the most relevance. Only provide the score nothing else. No extra information or explaination.", "relevance_score"

def get_contextual_metrics():
    return "{}\nHere is the response for the above scenario - {}\nCan you evaluate how well the contextual information of the above scenario is being utilized in the above response. Rate on a scale of 0-10, where 0 is for lowest level of contextual information, while 10 for the most. Only provide the score nothing else. No extra information or explaination.", "contextual_score"

def get_empathy_metrics():
    return "{}\nHere is the response for the above scenario - {}\nCan you evaluate how well this response empathetically to the victims's emotions and concerns. Rate on a scale of 0-10, where 0 is for lowest level of empathy, while 10 for the most. Only provide the score nothing else. No extra information or explaination.", "empathy_score"

for state in states:
    with open('../datadump/{}.json'.format(state), 'r') as openfile:
        json_object = json.load(openfile)
    response_list = []
    for data in json_object:
        question = data['Question']
        answer = data['Response']
        help_content, help_contents_metrices = get_help_metrics()
        help_content = help_content.format(question, answer)
        response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': help_content}])
        result = response['message']['content']
        response_list.append({'Question': question, 'Response': answer, help_contents_metrices: result})
        human_content, human_contents_metrices = get_humanlike_metrics()
        human_content = human_content.format(question, answer)
        response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': human_content}])
        result = response['message']['content']
        response_list.append({'Question': question, 'Response': answer, human_contents_metrices: result})
        relevance_content, relevance_contents_metrices = get_relevance_metrics()
        relevance_content = relevance_content.format(question, answer)
        response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': relevance_content}])
        result = response['message']['content']
        response_list.append({'Question': question, 'Response': answer, relevance_contents_metrices: result})
        contextual_content, contextual_contents_metrices = get_contextual_metrics()
        contextual_content = contextual_content.format(question, answer)
        response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': contextual_content}])
        result = response['message']['content']
        response_list.append({'Question': question, 'Response': answer, contextual_contents_metrices: result})
        empathy_content, empathy_contents_metrices = get_empathy_metrics()
        empathy_content = empathy_content.format(question, answer)
        response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': empathy_content}])
        result = response['message']['content']
        response_list.append({'Question': question, 'Response': answer, empathy_contents_metrices: result})
    with open("../datadump/{}_new_scoring.json".format(state), "w") as f:
        json.dump(response_list, f)
    with open("../datadump/{}_new_scoring.json".format(state), 'r') as openfile:
        json_object = json.load(openfile)
    final = ''
    for values in json_object:
        final = final + '\nQuestion : - {}\n\nAnswer : - {}\n\nHelpfulness : - {}\n\nHuman-Likeness : - {}\n\nRelevance : - {}\n\nContextual Information : - {}\n\nEmpathy : - {}\n'.format(values['Question'], values['Response'], values['helpful_score'], values['humanlike_score'], values['relevance_score'], values['contextual_score'], values['empathy_score'])
    with open("../datadump/{}_new_scoring.txt".format(state), "w", encoding="utf-8") as f:
        f.write(final)