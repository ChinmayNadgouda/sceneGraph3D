import json
from openai import OpenAI
import os
import base64
import torch
from PIL import Image
import numpy as np
import cv2
import ast
import re

from transformers import PreTrainedModel, AutoProcessor, LlavaNextProcessor

system_prompt_1 = '''
You are an agent specialized in describing the spatial relationships between objects in an annotated image.

You will be provided with an annotated image and a list of labels for the annotations. Your task is to determine the spatial relationships between the annotated objects in the image, and return a list of these relationships in the correct list of tuples format as follows:
[("object1", "spatial relationship", "object2"), ("object3", "spatial relationship", "object4"), ...]

Your options for the spatial relationship are "on top of" and "next to".

For example, you may get an annotated image and a list such as 
["cup 3", "book 4", "clock 5", "table 2", "candle 7", "music stand 6", "lamp 8"]

Your response should be a description of the spatial relationships between the objects in the image. 
An example to illustrate the response format:
[("book 4", "on top of", "table 2"), ("cup 3", "next to", "book 4"), ("lamp 8", "on top of", "music stand 6")]
'''

'''
You are an agent specialized in identifying and describing objects that are placed "on top of" each other in an annotated image. You always output a list of tuples that describe the "on top of" spatial relationships between the objects, and nothing else. When in doubt, output an empty list.

When provided with an annotated image and a corresponding list of labels for the annotations, your primary task is to determine and return the "on top of" spatial relationships between the annotated objects. Your responses should be formatted as a list of tuples, specifically highlighting objects that rest on top of others, as follows:
[("object1", "on top of", "object2"), ...]
'''

# Only deal with the "on top of" relation
system_prompt_only_top = '''
You are an agent specializing in identifying the physical and spatial relationships in annotated images for 3D mapping.

In the images, each object is annotated with a bright numeric id (i.e. a number) and a corresponding colored contour outline. Your task is to analyze the images and output a list of tuples describing the physical relationships between objects. Format your response as follows: [("1", "relation type", "2"), ...]. When uncertain, return an empty list.

Note that you are describing the **physical relationships** between the **objects inside** the image.

You will also be given a text list of the numeric ids of the objects in the image. The list will be in the format: ["1: name1", "2: name2", "3: name3" ...], only output the physical relationships between the objects in the list.

The relation types you must report are:
- phyically placed on top of: ("object x", "on top of", "object y") 
- phyically placed underneath: ("object x", "under", "object y") 

An illustrative example of the expected response format might look like this:
[("object 1", "on top of", "object 2"), ("object 3", "under", "object 2"), ("object 4", "on top of", "object 3")]. Do not put the names of the objects in your response, only the numeric ids.

Do not include any other information in your response. Only output a parsable list of tuples describing the given physical relationships between objects in the image.
'''

# For captions
system_prompt_captions = '''
You are an agent specializing in accurate captioning objects in an image.

In the images, each object is annotated with a bright numeric id (i.e. a number) and a corresponding colored contour outline. Your task is to analyze the images and output in a structured format, the captions for the objects.

You will also be given a text list of the numeric ids and names of the objects in the image. The list will be in the format: ["1: name1", "2: name2", "3: name3" ...]

The names were obtained from a simple object detection system and may be inaacurate.

Your response should be in the format of a list of dictionaries, where each dictionary contains the id, name, and caption of an object. Your response will be evaluated as a python list of dictionaries, so make sure to format it correctly. An example of the expected response format is as follows:
[
    {"id": "1", "name": "object1", "caption": "concise description of the object1 in the image"},
    {"id": "2", "name": "object2", "caption": "concise description of the object2 in the image"},
    {"id": "3", "name": "object3", "caption": "concise description of the object3 in the image"}
    ...
]

And each caption must be a concise description of the object in the image.
'''

system_prompt_consolidate_captions = '''
You are an agent specializing in consolidating multiple captions for the same object into a single, clear, and accurate caption.

You will be provided with several captions describing the same object. Your task is to analyze these captions, identify the common elements, remove any noise or outliers, and consolidate them into a single, coherent caption that accurately describes the object.

Ensure the consolidated caption is clear, concise, and captures the essential details from the provided captions.

Here is an example of the input format:
[
    {"id": "3", "name": "cigar box", "caption": "rectangular cigar box on the side cabinet"},
    {"id": "9", "name": "cigar box", "caption": "A small cigar box placed on the side cabinet."},
    {"id": "7", "name": "cigar box", "caption": "A small cigar box is on the side cabinet."},
    {"id": "8", "name": "cigar box", "caption": "Box on top of the dresser"},
    {"id": "5", "name": "cigar box", "caption": "A cigar box placed on the dresser next to the coffeepot."},
]

Your response should be a JSON object with the format:
{
    "consolidated_caption": "A small rectangular cigar box on the side cabinet."
}

Do not include any additional information in your response.
'''

system_prompt_llava16 = '''
You are an agent specialized in describing the spatial relationships between objects in an annotated image.

You will be provided with an annotated image and a list of labels for the annotations. Your task is to determine the spatial relationships between the annotated objects in the image, and return a list of these relationships in the correct list of tuples format as follows:
[("object1", "spatial relationship", "object2"), ("object3", "spatial relationship", "object4"), ...]
To consider the objects of interest, they must have a annotatted border with color. While forming the spatial realtionships just consider the object within this colored annotated border and ignore the text.
Do not consider spatial realtionships between objects which are too far away.
Your options for the spatial relationship are "above", "under" and "next to". 

For example, you may get an annotated image and a list such as 
["3: cup", "4: book", "5: clock", "7: candle", "6: music stand", "8: lamp"]

Your response should be a description of the spatial relationships between the objects in the image. 
An example to illustrate the response format (but don't include them in your answer):
[("4", "above", "6"), ("3", "next to", "4"), ("8", "under", "6")] Do not include these samples/examples in the final answer

Do not repeat the spatial relationships and limit the number of tuples to only 20 or less. I do not want the spatial realtionships to be repeated. Please limit them to 20 or less only. I want the list length to be only 20 or less. Do not put the names of the objects in your response, only the numeric ids.

Do not include any other information in your response. Only output a parsable list of tuples describing the given physical relationships between objects in the image.
'''
system_prompt = system_prompt_only_top

# gpt_model = "gpt-4-vision-preview"
gpt_model = "gpt-4o-2024-05-13"

def get_openai_client():
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY')
    )
    return client

# Function to encode the image as base64
def encode_image_for_openai(image_path: str, resize = False, target_size: int=512):
    print(f"Checking if image exists at path: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if not resize:
        # Open the image
        print(f"Opening image from path: {image_path}")
        with open(image_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
            print("Image encoded in base64 format.")
        return encoded_image
    
    print(f"Opening image from path: {image_path}")
    with Image.open(image_path) as img:
        # Determine scaling factor to maintain aspect ratio
        original_width, original_height = img.size
        print(f"Original image dimensions: {original_width} x {original_height}")
        
        if original_width > original_height:
            scale = target_size / original_width
            new_width = target_size
            new_height = int(original_height * scale)
        else:
            scale = target_size / original_height
            new_height = target_size
            new_width = int(original_width * scale)

        print(f"Resized image dimensions: {new_width} x {new_height}")

        # Resizing the image
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        print("Image resized successfully.")
        
        # Convert the image to bytes and encode it in base64
        with open("temp_resized_image.jpg", "wb") as temp_file:
            img_resized.save(temp_file, format="JPEG")
            print("Resized image saved temporarily for encoding.")
        
        # Open the temporarily saved image for base64 encoding
        with open("temp_resized_image.jpg", "rb") as temp_file:
            encoded_image = base64.b64encode(temp_file.read()).decode('utf-8')
            print("Image encoded in base64 format.")
        
        # Clean up the temporary file
        os.remove("temp_resized_image.jpg")
        print("Temporary file removed.")

    return encoded_image

def consolidate_captions(client: OpenAI, captions: list):
    # Formatting the captions into a single string prompt
    captions_text = "\n".join([f"{cap['caption']}" for cap in captions if cap['caption'] is not None])
    user_query = f"Here are several captions for the same object:\n{captions_text}\n\nPlease consolidate these into a single, clear caption that accurately describes the object."

    messages = [
        {
            "role": "system",
            "content": system_prompt_consolidate_captions
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    consolidated_caption = ""
    try:
        response = client.chat.completions.create(
            model=f"{gpt_model}",
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        consolidated_caption_json = response.choices[0].message.content.strip()
        consolidated_caption = json.loads(consolidated_caption_json).get("consolidated_caption", "")
        print(f"Consolidated Caption: {consolidated_caption}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        consolidated_caption = ""

    return consolidated_caption
    
def extract_list_of_tuples(text: str):
    # Pattern to match a list of tuples, considering a list that starts with '[' and ends with ']'
    # and contains any characters in between, including nested lists/tuples.
    text = text.replace('\n', ' ')
    pattern = r'\[.*?\]'
    
    # Search for the pattern in the text
    match = re.search(pattern, text)
    if match:
        # Extract the matched string
        list_str = match.group(0)
        try:
            # Convert the string to a list of tuples
            result = ast.literal_eval(list_str)
            if isinstance(result,tuple):
                result = ast.literal_eval('['+list_str+']')
            if isinstance(result, list):  # Ensure it is a list
                return result
        except (ValueError, SyntaxError):
            # Handle cases where the string cannot be converted
            print("Found string cannot be converted to a list of tuples.")
            return []
    else:
        # No matching pattern found
        print("No list of tuples found in the text.")
        return []
    
def vlm_extract_object_captions(text: str):
    # Replace newlines with spaces for uniformity
    text = text.replace('\n', ' ')
    
    # Pattern to match the list of objects
    pattern = r'\[(.*?)\]'
    
    # Search for the pattern in the text
    match = re.search(pattern, text)
    if match:
        # Extract the matched string
        list_str = match.group(0)
        try:
            # Try to convert the entire string to a list of dictionaries
            result = ast.literal_eval(list_str)
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            # If the whole string conversion fails, process each element individually
            elements = re.findall(r'{.*?}', list_str)
            result = []
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
    
def get_obj_rel_from_image_gpt4v(client: OpenAI, image_path: str, label_list: list):
    # Getting the base64 string
    base64_image = encode_image_for_openai(image_path)
    
    global system_prompt
    global gpt_model
    
    user_query = f"Here is the list of labels for the annotations of the objects in the image: {label_list}. Please describe the spatial relationships between the objects in the image."
    
    
    vlm_answer = []
    try:
        response = client.chat.completions.create(
            model=f"{gpt_model}",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt_only_top
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": user_query
                }
            ]
        )
        
        vlm_answer_str = response.choices[0].message.content
        print(f"Line 113, vlm_answer_str: {vlm_answer_str}")
        
        vlm_answer = extract_list_of_tuples(vlm_answer_str)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"Setting vlm_answer to an empty list.")
        vlm_answer = []
    print(f"Line 68, user_query: {user_query}")
    print(f"Line 97, vlm_answer: {vlm_answer}")
    
    
    return vlm_answer

    
def get_obj_captions_from_image_gpt4v(client: OpenAI, image_path: str, label_list: list):
    # Getting the base64 string
    base64_image = encode_image_for_openai(image_path)
    
    global system_prompt
    

    user_query = f"Here is the list of labels for the annotations of the objects in the image: {label_list}. Please accurately caption the objects in the image."
    
    messages=[
        {
            "role": "system",
            "content": system_prompt_captions
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
    
    
    vlm_answer_captions = []
    try:
        response = client.chat.completions.create(
            model=f"{gpt_model}",
            messages=messages
        )
        
        vlm_answer_str = response.choices[0].message.content
        print(f"Line 113, vlm_answer_str: {vlm_answer_str}")
        
        vlm_answer_captions = vlm_extract_object_captions(vlm_answer_str)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"Setting vlm_answer to an empty list.")
        vlm_answer_captions = []
    print(f"Line 68, user_query: {user_query}")
    print(f"Line 97, vlm_answer: {vlm_answer_captions}")
    
    
    return vlm_answer_captions


def consolidate_captions_llava(client: (PreTrainedModel,AutoProcessor), captions: list):
    # Formatting the captions into a single string prompt
    captions_text = "\n".join([f"{cap['caption']}" for cap in captions if cap['caption'] is not None])
    user_prompt = f"Here are several captions for the same object:\n{captions_text}\n\nPlease consolidate these into a single, clear caption that accurately describes the object."
    global system_prompt_consolidate_captions
    consolidated_caption = ""
    try:
        model = client[0]
        processor = client[1]
        prompt = f"{system_prompt_consolidate_captions} USER: \n{user_prompt} ASSISTANT: "
        inputs = processor(prompt, return_tensors='pt')
        output = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
        consolidated_caption_json = processor.decode(output[0][2:], skip_special_tokens=True)
        consolidated_caption = json.loads(consolidated_caption_json).get("consolidated_caption", "")
        print(f"Consolidated Caption: {consolidated_caption}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        consolidated_caption = ""

    return consolidated_caption

def get_obj_rel_from_image_llava(client: (PreTrainedModel, AutoProcessor), image_path: str, labels: list, depth_path = None, det_exp_vis_path = None):
    # Getting the base64 string
    user_prompt = f"Here is the list of labels for the annotations of the objects in the image: {labels}. Please describe the spatial relationships ( on top of, under, above, in, behind, in front of and next to ) between the objects in the image The top image is colored and has annotations and the bottom image is depth of the top."
    final_image_path = stitch_images(depth_path,image_path,det_exp_vis_path)
    print("******Stitched Image of depth and annotation for VLM - edges*******",final_image_path)
    raw_image = Image.open(final_image_path).convert('RGB')
    global system_prompt
    try:
        model = client[0]
        processor = client[1]
        prompt = f"{system_prompt} USER: <image>\n{user_prompt} ASSISTANT: "
        inputs = processor(prompt, raw_image, return_tensors='pt')
        output = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
        vlm_answer_str = processor.decode(output[0][2:], skip_special_tokens=True)
        print(f"Line 113, vlm_answer_str: {vlm_answer_str}")
        prompt2 = f"{system_prompt}USER:\n{user_prompt}ASSISTANT: "
        text_to_extract = extract_model_output(vlm_answer_str, prompt2)
        print(text_to_extract)
        vlm_answer = extract_list_of_tuples(text_to_extract)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"Setting vlm_answer to an empty list.")
        vlm_answer = []
    print(f"Line 68, user_query: {user_prompt}")
    print(f"Line 97, vlm_answer: {vlm_answer}")

    return vlm_answer


def get_obj_captions_from_image_llava(client: (PreTrainedModel, AutoProcessor), image_path: str, label_list: list, depth_path = None, det_exp_vis_path = None):
    # Getting the base64 string
    base64_image = encode_image_for_openai(image_path)
    # final_image_path = stitch_images(depth_path,image_path,det_exp_vis_path)
    # print("******Stitched Image of depth and annotation for VLM - captions*******",final_image_path)
    raw_image = Image.open(image_path).convert('RGB')

    user_prompt = f"Here is the list of labels for the annotations of the objects in the image: {label_list}. Please accurately caption the objects in the image."
    global system_prompt_captions

    vlm_answer_captions = []
    try:
        model = client[0]
        processor = client[1]
        prompt = f"{system_prompt_captions} USER: <image>\n{user_prompt} ASSISTANT: "
        inputs = processor(prompt, raw_image, return_tensors='pt')
        output = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
        vlm_answer_str = processor.decode(output[0][2:], skip_special_tokens=True)
        print(f"Line 113, vlm_answer_str: {vlm_answer_str}")
        prompt2 = f"{system_prompt_captions}USER:\n{user_prompt}ASSISTANT: "
        text_to_extract = extract_model_output(vlm_answer_str,prompt2)
        print(text_to_extract)
        vlm_answer_captions = vlm_extract_object_captions(text_to_extract)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"Setting vlm_answer to an empty list.")
        vlm_answer_captions = []
    print(f"Line 68, user_query: {user_prompt}")
    print(f"Line 97, vlm_answer: {vlm_answer_captions}")

    return vlm_answer_captions


def extract_model_output(full_response: str, prompt: str):
    # Remove the prompt from the start of the response
    if 'ASSISTANT: ' in full_response:
        # Extract the part of the response that comes after the prompt
        return full_response[len(prompt):].strip()
    return full_response.strip()


def consolidate_captions_llava_1_6_mistral(client: (PreTrainedModel, LlavaNextProcessor), captions: list):
    # Formatting the captions into a single string prompt
    captions_text = "\n".join([f"{cap['caption']}" for cap in captions if cap['caption'] is not None])
    user_prompt = f"Here are several captions for the same object:\n{captions_text}\n\nPlease consolidate these into a single, clear caption that accurately describes the object."
    global system_prompt_consolidate_captions
    consolidated_caption = ""
    try:
        model = client[0]
        processor = client[1]
        conversation = [
            {

            "role": "user",
            "content": [
                {"type": "text", "text": f"{system_prompt_consolidate_captions} USER: \n{user_prompt} ASSISTANT: "},
                {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(text=prompt, return_tensors="pt")

        # autoregressively complete prompt
        output = model.generate(**inputs, max_new_tokens=4000)
        consolidated_caption_json = processor.decode(output[0][2:], skip_special_tokens=True)
        consolidated_caption = json.loads(consolidated_caption_json).get("consolidated_caption", "")
        print(f"Consolidated Caption: {consolidated_caption}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        consolidated_caption = ""

    return consolidated_caption

def get_obj_rel_from_image_llava_1_6_mistral(client: (PreTrainedModel, LlavaNextProcessor), image_path: str, labels: list):
    # Getting the base64 string
    #torch.set_default_dtype(torch.float16)
    user_prompt = f"Here is the list of labels for the annotations of the objects in the image: {labels}. Please describe the spatial relationships between the objects in the image in the format: " + '''[("object 1", "on top of", "object 2"), ("object 3", "under", "object 2"), ("object 4", "on top of", "object 3")].'''
    user_prompt = f"Here is the list of labels for the annotations of the objects in the image: {labels}. You gave wrong realtions n the last answer. Please describe the spatial relationships between the objects in the image correctly and logically. Do not repeat the spatial relationships and limit the number of tuples to only 20 or less."
    raw_image = Image.open(image_path)
    global system_prompt
    try:
        model = client[0]
        processor = client[1]
        conversation = [
            {

                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt_llava16 + user_prompt},
                    {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(images=raw_image, text=prompt, return_tensors="pt")

        # autoregressively complete prompt
        output = model.generate(**inputs, max_new_tokens=4000, do_sample=False)
        vlm_answer_str = processor.decode(output[0][2:], skip_special_tokens=True)
        print(f"Line 113, vlm_answer_str: {vlm_answer_str}")
        prompt2 = f"{system_prompt}USER:\n{user_prompt}ASSISTANT: "
        text_to_extract = extract_model_output(vlm_answer_str, system_prompt_llava16 + user_prompt)
        print(text_to_extract)
        vlm_answer = extract_list_of_tuples(text_to_extract)
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"Setting vlm_answer to an empty list.")
        vlm_answer = []
    print(f"Line 68, user_query: {user_prompt}")
    print(f"Line 97, vlm_answer: {vlm_answer}")
    #default_dtype = torch.get_default_dtype()
    #torch.set_default_dtype(default_dtype)
    return vlm_answer


def get_obj_captions_from_image_llava_1_6_mistral(client: (PreTrainedModel, LlavaNextProcessor), image_path: str, label_list: list):
    # Getting the base64 string
    base64_image = encode_image_for_openai(image_path)
    raw_image = Image.open(image_path)

    user_prompt = f"Here is the list of labels for the annotations of the objects in the image: {label_list}. Please accurately caption the objects in the image."
    global system_prompt_captions

    vlm_answer_captions = []
    try:
        model = client[0]
        processor = client[1]
        conversation = [
            {

                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt_captions + user_prompt },
                    {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(images=raw_image, text=prompt, return_tensors="pt")

        # autoregressively complete prompt
        output = model.generate(**inputs, max_new_tokens=4000, do_sample=False)
        vlm_answer_str = processor.decode(output[0][2:], skip_special_tokens=True)
        print(f"Line 113, vlm_answer_str: {vlm_answer_str}")
        prompt2 = f"{system_prompt_captions}USER:\n{user_prompt}ASSISTANT: "
        text_to_extract = extract_model_output(vlm_answer_str,system_prompt_captions + user_prompt)
        print(text_to_extract)
        vlm_answer_captions = vlm_extract_object_captions(text_to_extract)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"Setting vlm_answer to an empty list.")
        vlm_answer_captions = []
    print(f"Line 68, user_query: {user_prompt}")
    print(f"Line 97, vlm_answer: {vlm_answer_captions}")

    return vlm_answer_captions
def get_affordance_label_from_list_for_given_obj(client: (PreTrainedModel, LlavaNextProcessor), object_class: str, affordance_list: list):
    prompt = """
    You are an agent specialized in identifying the correct affordance label of objects based on a list of affordance labels provided.
    Here is an example input, cabinet, ['pinch_pull','rotate','hook_turn','key_press','foot_push']
    You will be provided with an object class and a list of affordance labels for the object. Your task is to determine the most appropriate label. 
    Your response should be a only the affordance label in the following format ['pinch_pull']



    
    """
    user_prompt = 'Here is the object class,' + object_class + 'The affordance list are' +  affordance_list + 'Give me the most probable affordance label in rquested format.'
    conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": prompt + user_prompt}
        ]
        },
    ]

    try:
        model = client[0]
        processor = client[1]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(text=prompt, return_tensors="pt")

        # autoregressively complete prompt
        output = model.generate(**inputs, max_new_tokens=4000, do_sample=False)
        vlm_answer_str = processor.decode(output[0][2:], skip_special_tokens=True)
        print(f"Line 113, vlm_answer_str: {vlm_answer_str}")
        text_to_extract = extract_model_output(vlm_answer_str,prompt + user_prompt)
        print(text_to_extract)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"Setting vlm_answer to an empty list.")

def stitch_images(image_path1, image_path2, final_path):
    image1 = cv2.imread(image_path2)
    image2 = cv2.imread(image_path1)
    width = max(image1.shape[1], image2.shape[1])
    image1_resized = cv2.resize(image1, (width, int(image1.shape[0] * width / image1.shape[1])))
    image2_resized = cv2.resize(image2, (width, int(image2.shape[0] * width / image2.shape[1])))

    # Stitch images vertically
    stitched_image = cv2.vconcat([image1_resized, image2_resized])

    # Save the result
    cv2.imwrite((final_path / (image_path1.name + '_stitched_image')).with_suffix('.jpg'), stitched_image)
    
    return (final_path / (image_path1.name + '_stitched_image')).with_suffix('.jpg')


