
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
import torch
from PIL import Image
import requests
#model_id = "/home/student/ConceptGraph/llava_model/hub/models--llava-hf--llava-v1.6-mistral-7b-hf/snapshots/216670a16460adb7c41ce3e123ceb3859f73ab12"
model_id = "llava-hf/llava-v1.6-vicuna-13b-hf"
#model_id = "/home/student/.cache/huggingface/hub/models--llava-hf--llava-v1.6-vicuna-13b-hf"
processor = LlavaNextProcessor.from_pretrained(model_id)
bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_use_double_quant=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.bfloat16
)
model = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, quantization_config=bnb_config,
            use_flash_attention_2=True ) 
torch.set_default_dtype(torch.float16)
# model.to("cuda:0")
img_path = "sceneGraph3D/conceptgraph/scripts/frame000090annotated_for_vlm.jpg"
img_path2 = "sceneGraph3D/conceptgraph/scripts/image.jpg"
# prepare image and text prompt, using the appropriate prompt template
url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
#image = Image.open(requests.get(url, stream=True).raw)
image = Image.open(img_path)
image2 = Image.open(img_path2)
system_prompt_only_top = '''
You are an agent specialized in describing the spatial relationships between objects in an annotated image.

You will be provided with an annotated image and a list of labels for the annotations. Your task is to determine the spatial relationships between the annotated objects in the image, and return a list of these relationships in the correct list of tuples format as follows:
[("object1", "spatial relationship", "object2"), ("object3", "spatial relationship", "object4"), ...]
To consider the objects of interest, they must have a annotatted border with color. While forming the spatial realtionships just consider the object within this colored annotated border and ignore the text.
Your options for the spatial relationship are "above", "under" and "next to". 

For example, you may get an annotated image and a list such as 
["3: cup", "4: book", "5: clock", "7: candle", "6: music stand", "8: lamp"]

Your response should be a description of the spatial relationships between the objects in the image. 
An example to illustrate the response format (but don't include them in your answer):
[("4", "above", "6"), ("3", "next to", "4"), ("8", "under", "6")]

Do not repeat the spatial relationships and limit the number of tuples to only 20. I do not want the spatial realtionships to be repeated. Please limit them to 20 only. I want the list length to be only 20 or less. Do not put the names of the objects in your response, only the numeric ids.

Do not include any other information in your response. Only output a parsable list of tuples describing the given physical relationships between objects in the image.
'''


label_list = '["1: coffee kettle", "3: window", "8: cabinet", "14: book", "15: tissue box", "20: potted plant", "21: cabinet", "24: cabinet"]'
user_query = f"Here is the list of labels for the annotations of the objects in the image: {label_list}. You gave wrong realtions n the last answer. Please describe the spatial relationships between the objects in the image correctly and logically. Do not repeat the spatial relationships and limit the number of tuples to only 20."

# Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": system_prompt_only_top + user_query},
          {"type": "image"}
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

inputs = processor(images=image, text=prompt, return_tensors="pt")

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=500, do_sample=False)

print(processor.decode(output[0], skip_special_tokens=True))
