import json
import ollama
import pandas as pd

df = pd.read_csv('../dataset_domain/data.csv')
states = df['state'].unique()

for state in states:
    with open('../datadump/{}.json'.format(state), 'r') as openfile:
        json_object = json.load(openfile)
    response_list = []
    for data in json_object:
        question = data['Question']
        answer = data['Response']
        content = '{}\nHere is the response for the above scenario - {}\nCan you rate this information based on the help provided by the generated response on a scale of 0-10, where 0 is for lowest level of information, while 10 for the most information. Only provide the score nothing else. No extra information or explaination.'.format(
            question, answer
        )
        response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': content,},])
        result = response['message']['content']
        response_list.append({'Question': question, 'Response': answer, 'Scoring': result})
    with open("../datadump/{}_scoring.json".format(state), "w") as f:
        json.dump(response_list, f)
