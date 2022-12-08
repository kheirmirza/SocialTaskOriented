import json
import os
import pandas as pd
import numpy as np
import torch
import csv
import requests
from datasets import load_dataset
from sentence_transformers.util import semantic_search
import openai

prompt = '''
Convert the following sentence into their illocutionary force

Input:Please don't hurt me
Output:Begging

Input:Can you do me a favor
OutputRequest

Input:Would you like to visit me sometime
Output:Invitation

Input:I promise I'll be nice
Output:Promise
'''

def gpt3_completion(prompt, engine='text-davinci-002', temp=0.6, top_p=1.0, tokens=2000, freq_pen=0.25, pres_pen=0.0):
    max_retry = 5
    retry = 0
    openai.api_key = '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'
    while True:
        #try:
        response = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            temperature=temp,
            max_tokens=tokens,
            top_p=top_p,
            frequency_penalty=freq_pen,
            presence_penalty=pres_pen,)
        text = response['choices'][0]['text'].strip()
        return text
    

# Query database
def query(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()


def iloc(input):
    results = gpt3_completion(prompt + f'''
Input:{input[0].strip().encode(encoding='ASCII',errors='ignore').decode()}
Output:''', 
    temp=0, top_p=1, freq_pen=0, pres_pen=0)

    return results  



if __name__ == '__main__':
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    hf_token = "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"
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}

    # Load Embeds
    dataset = load_dataset("csv", data_files="embeddingsWeather.csv")
    dataset_embeddings = torch.from_numpy(dataset["train"].to_pandas().to_numpy()).to(torch.float)

    # Load Statements
    data = []
    with open('statements.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    # input = ["I would like to order some pizza"]

    while True:
        q = []
        user_input = input('Query: ')
        q = [user_input]

        # get iloc and decide whether you want to continue
        ilocutionaryForce = iloc(q)

        # get embeding for input
        output = query(q)
        query_embeddings = torch.FloatTensor(output)

        # get top 5 matches and format
        hits = semantic_search(query_embeddings, dataset_embeddings, top_k=5)
        statements = [data[hits[0][i]['corpus_id']] for i in range(len(hits[0]))]
        for h, s in zip(hits[0], statements):
            h['statement'] = s[0]

        selected = []

        for item in hits[0]:
            if item['score'] >= 0.8:
                selected.append(item)


        possible_ilocs = ['Request', 'Order', 'Command', 'Question', 'Prediction']

        if ilocutionaryForce in possible_ilocs and len(selected) > 0:

            for item in selected:
                print(item)
        
            print(f"Type: Task Oriented, Application: Weather, Iloc: {ilocutionaryForce}")
        else: 
            if ilocutionaryForce not in possible_ilocs:
                print(f"No match, Iloc: {ilocutionaryForce}, Possible ilocs: {possible_ilocs}")
            else:
                print(f"Iloc: {ilocutionaryForce}")
                print(f"Number of semantically similair statements: {len(selected)}")

# lovely weather today
# I'd like to order some pizza
# Will it be rainy tonight?

# conda activate
# python3 