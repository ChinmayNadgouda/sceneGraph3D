
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


model_id = "/home/gokul/ConceptGraphs/llava-v1.5-7b/models--llava-hf--llava-1.5-7b-hf/snapshots/fa3dd2809b8de6327002947c3382260de45015d4"


img_path = "frame000020annotated_for_vlm.jpg"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    load_in_4bit=True
)

processor = AutoProcessor.from_pretrained(model_id)

# raw_image = Image.open(requests.get(image_file, stream=True).raw)

# img_path = '/home/kuwajerw/new_local_data/new_record3d/ali_apartment/apt_scan_no_smooth_processed/exps/s_detections_stride1_69/vis/160_for_vlm.jpg'
raw_image = Image.open(img_path).convert('RGB')

labels = ['chair', 'pillow']#['power outlet 1', 'backpack 2', 'computer tower 3', 'poster 4', 'desk 5', 'picture 6', 'bowl 7', 'folded chair 8', 'trash bin 9', 'tissue box 10']

example_labels = []

# inputs = processor(prompt, raw_image, return_tensors='pt').to("cuda", torch.float16)
#system_prompt = "What follows is a chat between a human and an artificial intelligence assistant. The assistant always answers the question in the required format"
system_prompt = '''
1. A coffee kettle is placed on a coffee table.
2. A stool is located next to a lamp.
3. A pillow is placed on a sofa chair.
4. A potted plant is positioned near the closet door.
5. A stool is placed on the floor.
6. A lamp is situated on a table.
7. A window is present in the room.
8. A tissue box is located on a table.
9. A cabinet is placed in the room.
10. A power outlet is found on the wall.
11. A coffee table is situated in the room.
12. A light switch is located on the wall.
13. A plate is placed on a table.
14. A shelf is positioned on the wall.
15. A closet door is open in the room.
16. A lamp is placed on a table.
17. A stool is located on the floor.
18. A cabinet is placed in the room.
19. A coffee table is situated in the room.
20. A lamp is placed on a table.
21. A stool is located on the floor.
22. A cabinet is placed in the room.
23. A coffee table is situated in the room.
24. A lamp is placed on a table.
25. A stool is located on the floor.
26. A cabinet is placed in the room.
27. A coffee table is situated in the room.
28. A lamp is placed on a table.
29. A stool is located on the floor.
30. A cabinet is placed in the room.
31. A coffee table is situated in the room.
An illustrative example of the expected response format might look like this:
[("object 1", "on top of", "object 2"), ("object 3", "under", "object 2"), ("object 4", "on top of", "object 3")]. Do not put the names of the objects in your response, only the numeric ids.
Do not include any other information in your response. Only output a parsable list of tuples describing the given physical relationships between objects in the image.
'''

example_prompt = ""
#user_prompt = f"In this picture, there are these objects, {labels}. Are any of these  objects on top of one another in this picture?"
captions = [{'id':'1','name':'chair','caption': "A chair on the floor"}, {'id':'2','name':'chair','caption': "A chair with pillow on top"},  {'id':'2','name':'chair','caption': "A white colored chair"}]
captions_text = "\n".join([f"{cap['caption']}" for cap in captions if cap['caption'] is not None])
user_prompt = "Please describe the spatial relationships between the objects in the image according to required format.."
prompt = f"{system_prompt} USER: <image>\n{user_prompt} ASSISTANT:"
#print(f"Line 263, prompt: {prompt}")
inputs = processor(prompt, raw_image, return_tensors='pt')

output = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
resp = processor.decode(output[0],skip_special_tokens=True)
prompt2 = f"{system_prompt}\nUSER:\n{user_prompt}\nASSISTANT:"
extr = extract_model_output(resp ,prompt2)
print(resp)
vlm_answer_captions = vlm_extract_object_captions(extr)
print(f'test {vlm_answer_captions}')
k=1


