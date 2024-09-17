
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
#model_id = "/home/student/ConceptGraph/llava_model/hub/models--llava-hf--llava-v1.6-mistral-7b-hf/snapshots/216670a16460adb7c41ce3e123ceb3859f73ab12"
model_id = "llava-hf/llava-1.5-7b-hf"
#model_id = "/home/student/.cache/huggingface/hub/models--llava-hf--llava-v1.6-vicuna-13b-hf"
processor = LlavaNextProcessor.from_pretrained(model_id)

model = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True, load_in_4bit=True, use_flash_attention_2=True) 
#model.to("cuda:0")
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
Keep in mind, that the objects of interest have a border with color while forming the realtionships just consider the object within the border and ignore the annotation (1: box)
Your options for the spatial relationship are "on top of", "under",and "next to". And dont repeat or give self relationships

For example, you may get an annotated image and a list such as 
["3: cup", "4: book", "5: clock", "7: candle", "6: music stand", "8: lamp"]

Your response should be a description of the spatial relationships between the objects in the image. 
An example to illustrate the response format:
[("book 4", "on top of", "table 2"), ("cup 3", "next to", "book 4"), ("lamp 8", "on top of", "music stand 6")]

Dont include example in the output
'''


label_list = '["1: coffee kettle", "3: window", "8: cabinet", "14: book", "15: tissue box", "20: potted plant", "21: cabinet", "24: cabinet"]'
user_query = f"Here is the list of labels for the annotations of the objects in the image: {label_list}. Please describe the spatial relationships (on top of, under, above, in, behind, in front of and next to ) between the objects in the image. The top image is colored and has annotations and the bottom image is depth of the top."

# Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text":system_prompt_only_top + user_query},
          {"type": "image"}
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

inputs = processor(images=image2, text=prompt, return_tensors="pt")

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=10000)

print(processor.decode(output[0], skip_special_tokens=True))
