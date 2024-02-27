import json
import ollama
import re
import pandas as pd
import plotly.graph_objects as go

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

def scoring_states():
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
            contextual_content = get_contextual_metrics()
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

def get_score(data):
    value_list = re.findall(r'[-+]?(?:\d*\.*\d+)', data)
    # if len(value_list) > 1:
    #     print(value_list, data)
    if float(value_list[0]) >= 0 or float(value_list[0]) <= 10:
        return float(value_list[0])
    else:
        raise NotImplementedError

def radar_chat(df):
    categories = df['features']
    fig = go.Figure()
    df = df.drop(['features'], axis=1)
    for column in df.columns:
        fig.add_trace(go.Scatterpolar(r=df[column], theta=categories, name=column))        
    fig.show()
    fig.write_image("metrics.png")

def numeric_value():
    state_list = []
    data = {}
    for state in states:
        state_list = []
        sum_helpful_score = 0
        sum_humanlike_score = 0
        sum_relevance_score = 0
        sum_contextual_score = 0
        sum_empathy_score = 0
        with open("../datadump/{}_new_scoring.json".format(state), 'r') as openfile:
            json_object = json.load(openfile)
        for values in json_object:
            try:
                helpful_score = get_score(values['helpful_score'])
                humanlike_score = get_score(values['humanlike_score'])
                relevance_score = get_score(values['relevance_score'])
                contextual_score = get_score(values['contextual_score'])
                empathy_score = get_score(values['empathy_score'])
                sum_helpful_score = sum_helpful_score + helpful_score
                sum_humanlike_score = sum_humanlike_score + humanlike_score
                sum_relevance_score = sum_relevance_score + relevance_score
                sum_contextual_score = sum_contextual_score + contextual_score
                sum_empathy_score = sum_empathy_score + empathy_score
                # print(helpful_score, humanlike_score, relevance_score, contextual_score, empathy_score)
            except Exception as e:
                print("Exception Occured : - ".format(e))
        state_list.append(sum_helpful_score/len(json_object))
        state_list.append(sum_humanlike_score/len(json_object))
        state_list.append(sum_relevance_score/len(json_object))
        state_list.append(sum_contextual_score/len(json_object))
        state_list.append(sum_empathy_score/len(json_object))
        data[state] = state_list
        print("avg_helpful_score for {} is : - {}".format(state, sum_helpful_score/len(json_object)))
        print("avg_humanlike_score for {} is : - {}".format(state, sum_humanlike_score/len(json_object)))
        print("avg_relevance_score for {} is : - {}".format(state, sum_relevance_score/len(json_object)))
        print("avg_contextual_score for {} is : - {}".format(state, sum_contextual_score/len(json_object)))
        print("avg_empathy_score for {} is : - {}".format(state, sum_empathy_score/len(json_object)))
    data['features'] = ['helpfulness','humanlike','relevance','contextual','empathy']
    df = pd.DataFrame.from_dict(data)
    cf = df
    print(cf.columns)
    radar_chat(cf)

