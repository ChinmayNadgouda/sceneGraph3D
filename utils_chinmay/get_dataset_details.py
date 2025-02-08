
import os
from mistralai import Mistral

model = "mistral-large-latest"

client = Mistral(api_key='4hhDbtrs8XGj6XvsYP4YAm3qSl0zH2CO')
prompt = """You are an agent specialized in identifying the correct class label of an object part based on  the affordance label of the part and object class provided.
    Here is an example input, pinch_pull and  cabinet
    You will be provided with an object class and a affordance label for the part. Your task is to determine the most appropriate part name. 
    Your response should be a only the part label in the following format ['part_name']


Here is the object class, cabinet.
The affordance label for the part is pinch_pull
Give me the most probable part label in requested format."""
prompt = """You are an agent specialized in decribing at least five uses for a given part and its object.
    Here is an example input, handle and  cabinet
    You will be provided with an object class and a label for the part. Your task is to describe the five most appropriate tasks that can be done using the part. 
    Your response should be a only the tasks in the following format ['task1', 'task2', .... ]


Here is the object class, cabinet.
The label for the part is handle
Give me the tasks in requested format."""
chat_response = client.chat.complete(
    model= model,
    messages = [
        {
            "role": "user",
            "content": prompt,
        },
    ]
)
print(chat_response.choices[0].message.content)
exit()
import requests

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
    
output = query({
    "inputs": """[INST] You are an agent specialized in identifying the correct affordance label of objects based on a list of affordance labels provided.
    Here is an example input, cabinet, ['pinch_pull','rotate','hook_turn','key_press','foot_push']
    You will be provided with an object class and a list of affordance labels for the object. Your task is to determine the most appropriate label. 
    Your response should be a only the affordance label in the following format ['pinch_pull']


Here is the object class, cabinet.
The affordance list are  ['pinch_pull','rotate','hook_turn','key_press','foot_push']
Give me the most probable affordance label in requested format.
\n[/INST]""",
    "parameters": {
        "return_full_text": False
    }
})

print(output)
# from ctransformers import AutoModelForCausalLM, AutoTokenizer  
# import torch 
# cache_dir = "/media/gokul/Elements1/huggingface_cache" 

# llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GGUF", model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf", model_type="mistral", gpu_layers=50, cache_dir=cache_dir)

# model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"  # Use a smaller model if necessary
# cache_dir = "/media/gokul/Elements1/huggingface_cache" 

# # Use BitsAndBytesConfig correctly
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True, 
#     bnb_4bit_compute_dtype=torch.float16,  # Set computation type
#     bnb_4bit_quant_type="nf4",  # Use NormalFloat4
#     bnb_4bit_use_double_quant=True,  # Enable double quantization for better memory,
#         llm_int8_enable_fp32_cpu_offload=True

# )

# # Load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)  

# # Load the model with quantization
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",
#     quantization_config=bnb_config,  # Use the correct config object
#     cache_dir=cache_dir
# )

# def ask_llm(question):  
#     inputs = tokenizer(question, return_tensors="pt").to("cuda")  
#     output = model.generate(**inputs, max_new_tokens=100)  
#     return tokenizer.decode(output[0], skip_special_tokens=True)  

# print(ask_llm("What is a cabinet handle used for?"))
# import os
# import json
# class DatasetDetails():
#     def __init__(self, data_path):
#         self.data_path = data_path
#         self.test_ids = None
#         self.train_ids = None
#         self.val_ids = None

#         self.test_ids = self.get_test_ids()
#         self.train_ids = self.get_train_ids()
#         self.val_ids = self.get_val_ids()
#         self.id_details = None

#         self.id_details = self._get_details()
#         self.directory_name = {
#             'tip_push':20,
#             'hook_turn':21,
#             'exclude':22,
#             'hook_pull':23,
#             'key_press':24,
#             'rotate':25,
#             'foot_push':26,
#             'unplug':27,
#             'plug_in':28,
#             'pinch_pull':29    
#         }
#         self.text_files_home = '/home/student/move/Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
#         pass

#     def get_test_ids(self):
#         test_ids = []
#         for dirpath, dirnames, filenames in os.walk(self.data_path+'/test/'):
#             for file in filenames:
#                 test_ids.append(file.split('.')[0])
        
#         return test_ids

#     def get_train_ids(self):
#         train_ids = []
#         for dirpath, dirnames, filenames in os.walk(self.data_path+'/train/'):
#             for file in filenames:
#                 train_ids.append(file.split('.')[0])
        
#         return train_ids

#     def get_val_ids(self):    
#         val_ids = []
#         for dirpath, dirnames, filenames in os.walk(self.data_path+'/validation/'):
#             for file in filenames:
#                 val_ids.append(file.split('.')[0])
        
#         return val_ids
#     def get_details_by_id(self, id):
#         return self.id_details[id]

#     def _get_details(self):
#         for root, dirs, files in os.walk(self.scenefun_path):
#             if len(dirs) > 2:
#                 for dir in dirs:
#                     if dir.startswith('4'):
#                         json_file_path = root+'/'+dir+'/'+dir+'_annotations.json'
#                         with open(json_file_path, 'r') as file:
#                             data = json.load(file)
                        
#                         annotations = data['annotations']
#                         for annotation in annotations:
#                             if(annotation['label'] in self.directory_name):
                                
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  
import torch
cache_dir = "/media/gokul/Elements1/huggingface_cache" 
# model_name_llm = "mistralai/Mistral-7B-v0.1"  # Ensure you have enough VRAM  
# tokenizer = AutoTokenizer.from_pretrained(model_name_llm, token='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', cache_dir=cache_dir)  
# model = AutoModelForCausalLM.from_pretrained(model_name_llm, torch_dtype=torch.float16, device_map="cuda:0", cache_dir=cache_dir)  
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.float16,  # Set computation type
    bnb_4bit_quant_type="nf4",  # Use NormalFloat4
    bnb_4bit_use_double_quant=True,  # Enable double quantization for better memory,
    llm_int8_enable_fp32_cpu_offload=True
)
model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"  
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token='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')  
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", quantization_config=bnb_config, cache_dir=cache_dir) 
question =  'For the object cabinet, what would be the name of its part/ functionally interactive element if its label is hook_pull'

inputs = tokenizer(question, return_tensors="pt").to("cuda")  
output = model.generate(**inputs, max_new_tokens=100)  
print('LLM answer:',tokenizer.decode(output[0], skip_special_tokens=True))



# (thesis) (base) gokul@gokul-OMEN-by-HP-Transcend-Gaming-Laptop-16-u1xxx:~/ConceptGraphs$ pip list | grep transformers
# transformers                      4.44.0
# (thesis) (base) gokul@gokul-OMEN-by-HP-Transcend-Gaming-Laptop-16-u1xxx:~/ConceptGraphs$ pip list | grep bitsandbytes
# bitsandbytes                      0.43.3
# (thesis) (base) gokul@gokul-OMEN-by-HP-Transcend-Gaming-Laptop-16-u1xxx:~/ConceptGraphs$ pip list | grep accelerate
# accelerate                        0.33.0
# (thesis) (base) gokul@gokul-OMEN-by-HP-Transcend-Gaming-Laptop-16-u1xxx:~/ConceptGraphs$ 