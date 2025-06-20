from openai import OpenAI, AzureOpenAI
import os
import requests
import re
import google.generativeai as palm
import json
import replicate

openai_models = ['gpt-3.5-turbo-0613', 'gpt-4-0613']
google_models = ['text-bison-001']
replicate_models = ['llama-2-70b-chat', 'mixtral-8x7b-instruct-v0.1']
# replicate_models = ['llama-2-70b-chat']
hf_models = ['pmc-llama-13b', 'medalpaca-13b', 'meditron-7b', \
             'meditron-7b-chat', 'meditron-70b', 'mixtral-8x7b-instruct-v0.1']
azure_openai_models = ['azure-gpt-4o']

with open("api_config.json", "r") as jsonfile:
    api_config = json.load(jsonfile)
class llm_model:
    def __init__(self, model_name):
        self.model_name = model_name
        if model_name in azure_openai_models:
            self.client = AzureOpenAI(
                azure_endpoint=api_config['azure_openai']["AZURE_OPENAI_ENDPOINT"],
                api_key=api_config['azure_openai']["AZURE_OPENAI_API_KEY"],
                api_version=api_config['azure_openai']["OPENAI_API_VERSION"]
            )
            self.deployment_name = api_config['azure_openai']["DEPLOYMENT_NAME"]
            
        if self.model_name in openai_models:
            self.api_key = api_config['OpenAI']["API_KEY"]
            self.client = OpenAI(api_key=self.api_key)
            
        if self.model_name in google_models:
            self.api_key = api_config['Google']["API_KEY"]
            palm.configure(api_key=self.api_key)
        
        if self.model_name in replicate_models:
            self.api_key = api_config['Replicate']["API_KEY"]
            os.environ["REPLICATE_API_TOKEN"] = self.api_key
        
        if self.model_name in hf_models:
            self.api_url = api_config[self.model_name]["API_URL"]
            self.api_key = api_config[self.model_name]["API_KEY"]

    def query_model(self, prompt):
        # AZURE OPENAI MODELS
        if self.model_name in azure_openai_models:
            response = self._query_azure_openai(prompt)
            
        # OPENAI MODELS
        if self.model_name in openai_models:
            response = self._query_openai(prompt)
        
        # GOOGLE MODELS
        if self.model_name == 'text-bison-001':
            response = self._query_google(prompt)
        
        # REPLICATE MODELS
        if self.model_name == 'llama-2-70b-chat':
            response = self._query_replicate(prompt, max_new_tokens=40)
        
        if self.model_name == 'mixtral-8x7b-instruct-v0.1':
            response = self._query_replicate(prompt, max_new_tokens=40)
            # response = self._query_hf(prompt, max_new_tokens=40)

        # HUGGINGFACE MODELS            
        if self.model_name == 'pmc-llama-13b':
            response = self._query_hf(prompt, max_new_tokens=40)

        if self.model_name == "medalpaca-13b":
            response = self._query_hf(prompt, max_new_tokens=40)
        
        if self.model_name == "meditron-7b-chat":
            response = self._query_hf(prompt, max_new_tokens=40)

        if self.model_name == "meditron-7b":
            response = self._query_hf(prompt, max_new_tokens=40)

        if self.model_name == 'meditron-70b':
            response = self._query_hf(prompt, max_new_tokens=40)
        
        if len(response) == 0:
            response = "NR"
            return response
        
        # If response starts with a space, remove it
        if response[0] == ' ':
            response = response[1:]
        
        return response
    
    def _query_azure_openai(self, prompt):
        completion = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[{"role": "system", "content": "You are a medical GPT model tasked with making clinical decisions for research purposes only. You must respond with only a single letter and nothing more."},
                     {"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.1
        )
        response = completion.choices[0].message.content
        # Ensure we return only a single letter
        response = response.strip()
        if len(response) > 0 and response[0].isalpha():
            return response[0].upper()
        return "NR"

    def _query_openai(self, prompt):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=2048,
            messages=[{"role": "system", "content": prompt}])
        response = completion.choices[0].message.content
        return response

    def _query_google(self, prompt):
        completion = palm.generate_text(
            model=f'models/{self.model_name}',
            prompt=prompt,
        )
        response = completion.result

        if response is None:
            response = "NR"
        return response
    
    def _query_replicate(self, prompt, max_new_tokens=100):
        if self.model_name == 'llama-2-70b-chat':
            url = "meta/llama-2-70b-chat"
            output = replicate.run(url, input={"prompt": prompt, "max_new_tokens": max_new_tokens,"system_prompt": "answer"})
        elif self.model_name == 'mixtral-8x7b-instruct-v0.1':
            url = "mistralai/mixtral-8x7b-instruct-v0.1:cf18decbf51c27fed6bbdc3492312c1c903222a56e3fe9ca02d6cbe5198afc10"
        
            output = replicate.run(url, input={"prompt": prompt, "max_new_tokens": max_new_tokens})
        response = ''.join(output)
        
        return response

    def _query_hf(self, prompt, max_new_tokens=400):
        def query(payload):
            response = requests.post(self.api_url, headers=headers, json=payload)
            return response.json()
        
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        output = query({
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
            }
        })
        try:
            return output[0]['generated_text']
        except:
            # Print the error message
            print(output)
            return "NR"
