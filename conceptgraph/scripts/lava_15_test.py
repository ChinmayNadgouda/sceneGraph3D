
# WORKS, LAVA 1.5 13B, 4 BIT PRESCSION
import requests
from PIL import Image

import ast
import re
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

def extract_model_output(full_response: str, prompt: str):
    # Remove the prompt from the start of the response
    if 'ASSISTANT: ' in full_response:
        # Extract the part of the response that comes after the prompt
        return full_response[len(prompt):].strip()
    return full_response.strip()

def vlm_extract_object_captions(text: str):
    # Replace newlines with spaces for uniformity
    text = text.replace('\n', '_')

    # Pattern to match the list of objects
    pattern = r'\[(.*?)\]'

    # Search for the pattern in the text
    match = re.search(pattern, text)
    if match:
        print('akgkaldsjfklajdsfjalk;sdhgkla;shd;klgadhgklahdkgl')
        # Extract the matched string
        print(match)
        list_str = match.group(1)
        print(131313,list_str)
        try:
            # Try to convert the entire string to a list of dictionaries
            result = ast.literal_eval(list_str)
            print(12313,result)
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            # If the whole string conversion fails, process each element individually
            elements = re.findall(r'{.*?}', list_str)
            result = []
            print(12312313,elements)
            for element in elements:
                try:
                    obj = ast.literal_eval(element)
                    if isinstance(obj, dict):
                        result.append(obj)
                except (ValueError, SyntaxError):
                    print(f"Error processing element: {element}")
            return result
    else:
        # No matching pattern found
        print("No list of objects found in the text.")
        return []



# model_id = "/home/student/ConceptGraph/llava_model/hub/models--llava-hf--llava-v1.6-mistral-7b-hf/snapshots/216670a16460adb7c41ce3e123ceb3859f73ab12"
# #model_id = 'llava-hf/llava-v1.6-mistral-7b-hf'
model_id = "llava-hf/llava-1.5-7b-hf"

img_path = "sceneGraph3D/conceptgraph/scripts/frame000090annotated_for_vlm.jpg"
img_path2 = "sceneGraph3D/conceptgraph/scripts/image.jpg"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    load_in_4bit=True
)

processor = AutoProcessor.from_pretrained(model_id)

# raw_image = Image.open(requests.get(image_file, stream=True).raw)

# img_path = '/home/kuwajerw/new_local_data/new_record3d/ali_apartment/apt_scan_no_smooth_processed/exps/s_detections_stride1_69/vis/160_for_vlm.jpg'
raw_image = Image.open(img_path2).convert('RGB')

labels = ['chair', 'pillow']#['power outlet 1', 'backpack 2', 'computer tower 3', 'poster 4', 'desk 5', 'picture 6', 'bowl 7', 'folded chair 8', 'trash bin 9', 'tissue box 10']

example_labels = []

system_prompt_only_top = '''
You are an agent specialized in describing the spatial relationships between objects in an annotated image.

You will be provided with an annotated image and a list of labels for the annotations. Your task is to determine the spatial relationships between the annotated objects in the image, and return a list of these relationships in the correct list of tuples format as follows:
[("object1", "spatial relationship", "object2"), ("object3", "spatial relationship", "object4"), ...]
Your options for the spatial relationship are "on top of", "under",and "next to". 

For example, you may get an annotated image and a list such as 
["3: cup", "4: book", "5: clock", "7: candle", "6: music stand", "8: lamp"]

Your response should be a description of the spatial relationships between the objects in the image. 
An example to illustrate the response format:
[("book 4", "on top of", "table 2"), ("cup 3", "next to", "book 4"), ("lamp 8", "on top of", "music stand 6")]
'''
label_list = '["1: coffee kettle", "3: window", "8: cabinet", "14: book", "15: tissue box", "20: potted plant", "21: cabinet", "24: cabinet"]'

example_prompt = ""
#user_prompt = f"In this picture, there are these objects, {labels}. Are any of these  objects on top of one another in this picture?"
captions = [{'id':'1','name':'chair','caption': "A chair on the floor"}, {'id':'2','name':'chair','caption': "A chair with pillow on top"},  {'id':'2','name':'chair','caption': "A white colored chair"}]
captions_text = "\n".join([f"{cap['caption']}" for cap in captions if cap['caption'] is not None])
user_prompt = f"Here is the list of labels for the annotations of the objects in the image: {label_list}. Please describe the spatial relationships (on top of, under, above, in, behind, in front of and next to ) between the objects in the image. The top image is colored and has annotations and the bottom image is depth of the top."
prompt = f"{system_prompt_only_top} USER: <image>\n{user_prompt} ASSISTANT:"
#print(f"Line 263, prompt: {prompt}")
inputs = processor(prompt, images=raw_image, return_tensors='pt')

if 'image_sizes' in inputs:
    del inputs['image_sizes']
output = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
resp = processor.decode(output[0],skip_special_tokens=True)
prompt2 = f"{system_prompt_only_top}\nUSER:\n{user_prompt}\nASSISTANT:"
extr = extract_model_output(resp ,prompt2)
print(resp)
vlm_answer_captions = vlm_extract_object_captions(extr)
print(f'test {vlm_answer_captions}')
k=1