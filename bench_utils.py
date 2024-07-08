import os
import io
import json
import time
import yaml
import base64
import random
import regex as re
import requests
import hashlib
from PIL import Image
from io import BytesIO
from typing import Union
from mimetypes import guess_type
from pathlib import Path
from icecream import ic

# API setting constants
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"


OPENAI_MODEL_LIST = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-0613-verbose",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
)


temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
}


def encode_image(image:Image.Image) -> str:
    im_file = BytesIO()
    image.save(im_file, format="PNG")
    im_bytes = im_file.getvalue()
    im_64 = base64.b64encode(im_bytes).decode("utf-8")
    return json.dumps(im_64)

def convert_pil_to_base64(image):
    # convert pil image to base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode('utf-8')

def read_http_image_as_bytes(url):
    """Reads an image from a URL into bytes."""
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to retrieve image. Status code: {response.status_code}")

def model_name_to_id(model_name):
    return model_name.replace("/", "_")

def get_image_size(image:Image.Image):
    width, height = image.size

    # Create a BytesIO object
    img_byte_arr = io.BytesIO()

    # Save image to the BytesIO object
    image.save(img_byte_arr, format=image.format)

    # Get the size of the BytesIO object (in bytes)
    img_size_bytes = img_byte_arr.tell()

    # Convert bytes to megabytes
    img_size_mb = img_size_bytes / 1048576  # 1 MB = 1048576 bytes
    return img_size_mb

def load_question_categoeis(bench_name:str) -> dict:
    if bench_name == "vision_bench":
        import datasets
        # TODO: release category and domain types
        dataset = datasets.load_dataset("WildVision/wildvision-bench", "taxonmy")['test_with_taxnomy']
        question_ids = dataset["question_id"]
        question_categories = dataset["question_category"]
        categories = {}
        for question_id, question_category in zip(question_ids, question_categories):
            categories[question_id] = question_category
        print(f"Loaded {len(categories)} question ategories for {bench_name}")
        return categories
    else:
        raise ValueError("Invalid bench name")

def load_image_categoeis(bench_name:str) -> dict:
    if bench_name == "vision_bench":
        import datasets
        # TODO: release category and domain types
        dataset = datasets.load_dataset("WildVision/wildvision-bench", "taxonmy")['test_with_taxnomy']
        image_subdomains = dataset["image_domain"]
        image_ids = dataset["question_id"]
        categories = {}
        for image_id, image_subdomain in zip(image_ids, image_subdomains):
            categories[image_id] = image_subdomain
        print(f"Loaded {len(categories)} image categories for {bench_name}")
        return categories
    else:
        raise ValueError("Invalid bench name")
        

def load_model_answers(answer_dir, category=None):
    model_answers = {}
    if not os.path.exists(answer_dir):
        return model_answers
    for file in os.listdir(answer_dir):
        if file.endswith(".jsonl"):
            model = file.replace(".jsonl", "")
            model_answers[model] = {}
            with open(os.path.join(answer_dir, file), "r") as f:
                for line in f:
                    data = json.loads(line)
                    model_answers[model][data["question_id"]] = data
            print(f"Loaded {len(model_answers[model])} answers for {model}")
    return model_answers

def load_model_judgements(judgement_dir, SAMPLE_START=0, MAX_SAMPLE_BENCH_SIZE=1000):
    model_judgements = {}
    for file in os.listdir(judgement_dir):
        if file.endswith(".jsonl"):
            model = file.replace(".jsonl", "")
            model_judgements[model] = {}
            with open(os.path.join(judgement_dir, file), "r") as f:
                for line in f:
                    data = json.loads(line)
                    model_judgements[model][data["question_id"]] = data
            judge = data["judge"]
            print(f"Loaded {len(model_judgements[model])} judgements for {model}, judged by {judge}")
    return model_judgements

def hash_pil_image(pil_img):
    """Compute the SHA-256 hash of the contents of a PIL Image object.

    Args:
        pil_img (PIL.Image.Image): The PIL Image object.

    Returns:
        str: The hexadecimal SHA-256 hash of the image.
    """
    # Create a hash object
    sha256 = hashlib.sha256()

    # Convert the PIL Image to bytes
    img_byte_arr = BytesIO()
    pil_img.save(img_byte_arr, format='PNG')  # You can change 'PNG' to any format PIL supports

    # Update the hash with the image bytes
    sha256.update(img_byte_arr.getvalue())

    # Return the hexadecimal digest of the hash
    return sha256.hexdigest()

def openai_local_image_to_data_url(image:Union[str, Image.Image, Path]) -> str:
    if isinstance(image, Path) and image.exists() or isinstance(image, str) and os.path.exists(image):
        image_path = image
        # Guess the MIME type of the image based on the file extension
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'  # Default MIME type if none is found

        # Read and encode the image file
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{base64_encoded_data}"
        
    elif isinstance(image, Image.Image):
        dummy_path = f"temp.{image.format}"
        mime_type, _ = guess_type(dummy_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'
        # encode the image
        with BytesIO() as output:
            image.save(output, format=image.format)
            base64_encoded_data = base64.b64encode(output.getvalue()).decode('utf-8')
        return f"data:{mime_type};base64,{base64_encoded_data}"
    elif isinstance(image, str) and (image.startswith("http") or image.startswith("data:")):
        return image
    else:
        raise ValueError("Image must be a path to a local image, an image object, or a URL.")
    
def gemini_smart_process_image(image:Union[str, Image.Image, Path]) -> str:
    import google.ai.generativelanguage as glm
    if isinstance(image, Path) and image.exists() or isinstance(image, str) and os.path.exists(image):
        image_path = image
        # Guess the MIME type of the image based on the file extension
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'  # Default MIME type if none is found

        # Read and encode the image file
        image_bytes = Path(image_path).read_bytes()
        return glm.Part(
            inline_data=glm.Blob(
                mime_type=mime_type,
                data=image_bytes,
            )
        )
        
    elif isinstance(image, Image.Image):
        dummy_path = f"temp.{image.format}"
        mime_type, _ = guess_type(dummy_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'
        # encode the image
        with BytesIO() as output:
            image.save(output, format=image.format)
            image_bytes = output.getvalue()
        return glm.Part(
            inline_data=glm.Blob(
                mime_type=mime_type,
                data=image_bytes,
            )
        )
    elif isinstance(image, str) and (image.startswith("http") or image.startswith("data:")):
        if re.match(r"^data:image\/[a-zA-Z]*;base64,", image):
            mime_type = re.search(r"^data:image\/([a-zA-Z]*)", image).group(1)
            base64_encoded_data = re.sub(r"^data:image\/[a-zA-Z]*;base64,", "", image)
            image_bytes = base64.b64decode(base64_encoded_data)
            return glm.Part(
                inline_data=glm.Blob(
                    mime_type=f"image/{mime_type}",
                    data=image_bytes,
                )
            )
        elif re.match(r"^http", image):
            image_bytes = read_http_image_as_bytes(image)
            mime_type, _ = guess_type(image)
            if mime_type is None:
                mime_type = 'application/octet-stream'
            return glm.Part(
                inline_data=glm.Blob(
                    mime_type=mime_type,
                    data=image_bytes,
                )
            )
    else:
        raise ValueError("Image must be a path to a local image, an image object, or a URL.")
    
def anthropic_smart_process_image(image:Union[str, Image.Image, Path]) -> str:
    if isinstance(image, Path) and image.exists() or isinstance(image, str) and os.path.exists(image):
        image_path = image
        # Guess the MIME type of the image based on the file extension
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'  # Default MIME type if none is found

        # Read and encode the image file
        image_bytes = Path(image_path).read_bytes()
        return {
            "type": "base64",
            "data": base64.b64encode(image_bytes).decode('utf-8')
        }
        
    elif isinstance(image, Image.Image):
        dummy_path = f"temp.{image.format}"
        mime_type, _ = guess_type(dummy_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'
        
        # encode the image
        with BytesIO() as output:
            image.save(output, format=image.format)
            base64_encoded_data = base64.b64encode(output.getvalue()).decode('utf-8')
        return {
            "type": "base64",
            "media_type": mime_type,
            "data": base64_encoded_data
        }
            
    elif isinstance(image, str) and (image.startswith("http") or image.startswith("data:")):
        if re.match(r"^data:image\/[a-zA-Z]*;base64,", image):
            mime_type = re.search(r"^data:image\/([a-zA-Z]*)", image).group(1)
            base64_encoded_data = re.sub(r"^data:image\/[a-zA-Z]*;base64,", "", image)
            return {
                "type": "base64",
                "media_type": f"image/{mime_type}",
                "data": base64_encoded_data
            }
        elif re.match(r"^http", image):
            image_bytes = read_http_image_as_bytes(image)
            mime_type, _ = guess_type(image)
            if mime_type is None:
                mime_type = 'application/octet-stream'
            return {
                "type": "base64",
                "media_type": mime_type,
                "data": base64.b64encode(image_bytes).decode('utf-8')
            }
    else:
        raise ValueError("Image must be a path to a local image, an image object, or a URL.")

def get_endpoint(endpoint_list):
    if endpoint_list is None:
        return None
    assert endpoint_list is not None
    # randomly pick one
    api_dict = random.choices(
        endpoint_list
    )[0]
    return api_dict


# load config args from config yaml files
def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    return config_kwargs


def chat_completion_openai(model, messages, temperature, max_tokens, api_dict=None, is_yivl_api=False):
    if is_yivl_api:
        import openai
        from openai import OpenAI
        client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=os.getenv("YIVL_API_KEY"),
            base_url=os.getenv("YIVL_API_BASE")
        )
    else:
        import openai
        if api_dict:
            client = openai.OpenAI(
                base_url=api_dict["api_base"],
                api_key=api_dict["api_key"],
            )
        else:
            client = openai.OpenAI()
    
    openai_messages = []
    for message in messages:
        if message["role"] == "user" and isinstance(message["content"], list):
            
            has_image = any([x['type'] == 'image' for x in message["content"]])
            if has_image:
                openai_messages.append({
                    "role": "user",
                    "content": [],
                })
                for i in range(len(message["content"])):
                    if message["content"][i]["type"] == "image":
                        openai_messages[-1]["content"].append({
                            "type": "image_url",
                            "image_url": {
                                "url": openai_local_image_to_data_url(message["content"][i]["image"]),
                            },
                        })
                    else:
                        assert message["content"][i]["type"] == "text"
                        openai_messages[-1]["content"].append({
                            "type": "text",
                            "text": message["content"][i]["text"],
                        })
            else:
                assert len(message["content"]) == 1 and message["content"][0]["type"] == "text"
                openai_messages[-1]["content"].append({
                    "role": "user",
                    "content": message["content"][0]["text"],
                })
        else:
            openai_messages.append({
                "role": message["role"],
                "content": message["content"],
            })
    
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            # print(messages)
            completion = client.chat.completions.create(
                model=model,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens
                )
            if is_yivl_api:
                # return completion
                # YIVL API change
                output = completion.choices[0].message.content
            else:
                output = completion.choices[0].message.content
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            if "You uploaded an unsupported image." in str(e):
                print("Trying to reduce the image size")
                for message in messages:
                    if message["role"] == "user" and isinstance(message["content"], list):
                        for i in range(len(message["content"])):
                            if message["content"][i]["type"] == "image":
                                try:
                                    image_data = message["content"][i]["image"]
                                    if isinstance(image_data, str):
                                        mime_type, image_bytes = re.search(r"data:(.*);base64,(.*)", image_data).groups()
                                        image_bytes = base64.b64decode(image_bytes)
                                        img = Image.open(BytesIO(image_bytes))
                                        image_mega_bytes = get_image_size(img)  # Ensure this function exists and is correctly implemented
                                        if image_mega_bytes > 20:
                                            scale_factor = image_mega_bytes / 20
                                        else:
                                            scale_factor = 2
                                        resized_img = img.resize((int(img.width // scale_factor), int(img.height // scale_factor)))
                                        resized_img.format = img.format
                                        new_image = openai_local_image_to_data_url(resized_img)  # Ensure this function exists and is correctly implemented
                                        message["content"][i] = {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": new_image,
                                            },
                                        }
                                    else:
                                        print("Image data is not a string.")
                                except Exception as e:
                                    print(f"Failed to process image with error: {e}")
            else:
                print(type(e), e)
            break
        except KeyError:
            print(type(e), e)
            break
    
    return output


def chat_completion_openai_azure(model, messages, temperature, max_tokens, api_dict=None):
    import openai
    from openai import AzureOpenAI

    api_base = api_dict["api_base"]
    client = AzureOpenAI(
        azure_endpoint = api_base,
        api_key= api_dict["api_key"],
        api_version=api_dict["api_version"],
        timeout=240,
        max_retries=2
    )

    for message in messages:
        if message["role"] == "user" and isinstance(message["content"], list):
            has_image = any([x['type'] == 'image' for x in message["content"]])
            if has_image:
                for i in range(len(message["content"])):
                    if message["content"][i]["type"] == "image":
                        message["content"][i] = {
                            "type": "image_url",
                            "image_url": {
                                "url": openai_local_image_to_data_url(message["content"][i]["image"]),
                            },
                        }
                    else:
                        assert message["content"][i]["type"] == "text"
            else:
                assert len(message["content"]) == 1 and message["content"][0]["type"] == "text"
                message["content"] = message["content"][0]["text"]
                
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=42,
            )
            output = response.choices[0].message.content
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            if "You uploaded an unsupported image." in str(e):
                print("Trying to reduce the image size")
                for message in messages:
                    if message["role"] == "user" and isinstance(message["content"], list):
                        for i in range(len(message["content"])):
                            if message["content"][i]["type"] == "image":
                                # "data:{mime_type};base64,{base64_encoded_data}"
                                mime_type, image_bytes = re.search(r"data:(.*);base64,(.*)", message["content"][i]["image"]).groups()
                                image_bytes = base64.b64decode(image_bytes)
                                img = Image.open(BytesIO(image_bytes))
                                image_mega_bytes = get_image_size(img)
                                if image_mega_bytes > 20:
                                    scale_factor = image_mega_bytes / 20
                                else:
                                    scale_factor = 2
                                resized_img = img.resize((int(img.width // scale_factor), int(img.height // scale_factor)))
                                resized_img.format = img.format
                                new_image = openai_local_image_to_data_url(resized_img)
                                message["content"][i] = {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": new_image,
                                    },
                                }
            else:
                print(type(e), e)
            break
        except KeyError:
            print(type(e), e)
            break

    return output


def chat_completion_anthropic(model, messages, temperature, max_tokens, api_dict=None):
    import anthropic

    if api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["ANTHROPIC_API_KEY"]

    sys_msg = ""
    if messages[0]["role"] == "system":
        sys_msg = messages[0]["content"]
        messages = messages[1:]
        
    for message in messages:
        if message["role"] == "user" and isinstance(message["content"], list):
            has_image = any([x['type'] == 'image' for x in message["content"]])
            if has_image:
                for i in range(len(message["content"])):
                    if message["content"][i]["type"] == "image":
                        message["content"][i] = {
                            "type": "image",
                            "source": anthropic_smart_process_image(message["content"][i]["image"]),
                        }
                    else:
                        assert message["content"][i]["type"] == "text"
            else:
                assert len(message["content"]) == 1 and message["content"][0]["type"] == "text"
                message["content"] = message["content"][0]["text"]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            # print(sys_msg)
            c = anthropic.Anthropic(api_key=api_key)
            response = c.messages.create(
                model=model,
                messages=messages,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                max_tokens=max_tokens,
                temperature=temperature,
                system=sys_msg
            )
            output = response.content[0].text
            break
        except anthropic.APIError as e:
            if "image exceeds 5 MB maximum" in str(e):
                print("Image exceeds 5 MB maximum, trying to reduce the image size")
                for message in messages:
                    if message["role"] == "user" and isinstance(message["content"], list):
                        for i in range(len(message["content"])):
                            if message["content"][i]["type"] == "image":
                                img_bytes = base64.b64decode(message["content"][i]["source"]["data"])
                                img = Image.open(BytesIO(img_bytes))
                                image_mega_bytes = get_image_size(img)
                                if image_mega_bytes > 5:
                                    scale_factor = image_mega_bytes / 5
                                else:
                                    scale_factor = 2
                                resized_img = img.resize((int(img.width // scale_factor), int(img.height // scale_factor)))
                                resized_img.format = img.format
                                new_source = anthropic_smart_process_image(resized_img)
                                message["content"][i]["source"] = new_source
            else:
                print(type(e), e)
                break
    assert isinstance(output, str)
    return output

def https_image_service(image):
    from arena.constants import WEB_IMG_FOLDER
        
    WEB_IMG_URL_ROOT = os.getenv("WEB_IMG_URL_ROOT")
    import shortuuid
    img_id = shortuuid.uuid()
    
    # resize image is too large
    width, height = image.size
    if width > 1024 or height > 1024:
        image.thumbnail((1024, 1024))
    image.save(os.path.join(WEB_IMG_FOLDER, f"{img_id}.png"))
    media_url = f"{WEB_IMG_URL_ROOT}/{img_id}.png"
    return media_url

idefics2_model = None
idefics2_processor = None
def chat_completion_idefics2(model_name, text, image, temperature, top_p, max_tokens):
    # from text_generation import Client
    # import google.generativeai as genai
    # HF_API_TOKEN = os.getenv('HF_API_TOKEN')
    # API_URL = "https://api-inference.huggingface.co/models/HuggingFaceM4/idefics2-8b-chatty"
    
    # SYSTEM_PROMPT = "System: The following is a conversation between Idefics2, a highly knowledgeable and intelligent visual AI assistant created by Hugging Face, referred to as Assistant, and a human user called User. In the following interactions, User and Assistant will converse in natural language, and Assistant will do its best to answer User’s questions. Assistant has the ability to perceive images and reason about them, but it cannot generate images. Assistant was built to be respectful, polite and inclusive. It knows a lot, and always tells the truth. When prompted with an image, it does not make up facts.<end_of_utterance>\nAssistant: Hello, I'm Idefics2, Huggingface's latest multimodal assistant. How can I help you?<end_of_utterance>\n"
    # media_url = https_image_service(image) # dev: "https://raw.githubusercontent.com/huggingface/text-generation-inference/main/integration-tests/images/chicken_on_money.png"
    # # conv.set_media_url(media_url)
    # prompt = text #conv.to_idefics2_messages()
    # # ic(conv, prompt)
    # QUERY = f"User:![]({media_url}){prompt}<end_of_utterance>\nAssistant:"

    # client = Client(
    #     base_url=API_URL,
    #     headers={"x-use-cache": "0", "Authorization": f"Bearer {HF_API_TOKEN}"},
    #     timeout=60,
    # )
    # generation_args = {
    #     "max_new_tokens": max_tokens,
    #     "repetition_penalty": 1.1,
    #     "temperature": max(min(temperature, 1.0-1e-3), 1e-3),
    #     # "top_p": max(min(top_p, 1.0-1e-3), 1e-3),
    #     "do_sample": True if temperature > 1e-3 else False,
    # }
    # # generated_text = client.generate(prompt=SYSTEM_PROMPT + QUERY, **generation_args)
    # # data = {
    # #     "text": generated_text,
    # #     "error_code": 0,
    # # }
    # try:
    #     text = client.generate(prompt=SYSTEM_PROMPT + QUERY, **generation_args).generated_text
    # except Exception as e:
    #     print(type(e), e, "\n", SYSTEM_PROMPT + QUERY)
    #     text = API_ERROR_OUTPUT
    
    
    global idefics2_model, idefics2_processor
    from transformers import AutoModelForVision2Seq, AutoProcessor
    import torch
    if not idefics2_model:
        idefics2_model = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/idefics2-8b-chatty",
            torch_dtype=torch.float16, device_map="auto")
    if not idefics2_processor:
        idefics2_processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b-chatty")
        
    SYSTEM_PROMPT = "The following is a conversation between Idefics2, a highly knowledgeable and intelligent visual AI assistant created by Hugging Face, referred to as Assistant, and a human user called User. In the following interactions, User and Assistant will converse in natural language, and Assistant will do its best to answer User’s questions. Assistant has the ability to perceive images and reason about them, but it cannot generate images. Assistant was built to be respectful, polite and inclusive. It knows a lot, and always tells the truth. When prompted with an image, it does not make up facts.<end_of_utterance>"
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT},
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hello, I'm Idefics2, Huggingface's latest multimodal assistant. How can I help you?"},
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ]
        },
    ]
    # print(messages)
    prompt = idefics2_processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = idefics2_processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: v.to(idefics2_model.device) for k, v in inputs.items()}
    input_token_len = inputs["input_ids"].shape[1]
    # Generate
    generated_ids = idefics2_model.generate(**inputs, max_new_tokens=max_tokens)
    generated_texts = idefics2_processor.batch_decode(generated_ids[:, input_token_len:], skip_special_tokens=True)
    text = generated_texts[0]

    return text

def chat_completion_rekaflash(model_name, text, image, temperature, top_p, max_tokens, retry=5):
    import reka
    media_url = https_image_service(image)
    ic(media_url)
    # conv.set_media_url(media_url)
    # conv.set_media_type("image")
    # prompt = conv.to_reka_api_messages()
    gen_params = {
        "model": model_name,
        # "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_tokens,
    }
    retried = 0
    output = "$ERROR$"
    while retried < retry:
        try:
            response = reka.chat(
                conversation_history=[{
                    "type": "human",
                    "text": text,
                    "media_url": media_url,
                }],
                # media_type="image",
                request_output_len=gen_params["max_new_tokens"],
                temperature=gen_params["temperature"],
                # runtime_top_k=1024,
                runtime_top_p=gen_params["top_p"],
            )
            output = response["text"]
            return output
        except Exception as e:
            ic(e)
            retried += 1
            time.sleep(3)
    return output

def chat_completion_mistral(model, messages, temperature, max_tokens):
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
    from mistralai.exceptions import MistralException

    api_key = os.environ["MISTRAL_API_KEY"]
    client = MistralClient(api_key=api_key)

    prompts = [ChatMessage(role=message["role"], content=message["content"]) for message in messages]
    
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            chat_response = client.chat(
                model=model,
                messages=prompts,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = chat_response.choices[0].message.content
            break
        except MistralException as e:
            print(type(e), e)
            break

    return output

def chat_completion_gemini(model, messages, temperature, max_tokens):
    import google.generativeai as genai
    import google.ai.generativelanguage as glm
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

    safety_settings = [
        {
            "category": "HARM_CATEGORY_DANGEROUS",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]

    # Set up the model
    generation_config = {
        "temperature": temperature,
        "top_p": 1,
        "top_k": 1,
        # "max_output_tokens": max_tokens, # TODO: remove according to this: https://github.com/google-gemini/generative-ai-python/issues/170#issuecomment-1890987296
    }

    gemini_messages = []
    for message in messages:
        if message["role"] == "user":
            if isinstance(message["content"], list):
                gemini_messages.append({
                    "role": "user",
                    "parts": [],
                })
                has_image = any([x['type'] == 'image' for x in message["content"]])
                if has_image:
                    
                    for i in range(len(message["content"])):
                        if message["content"][i]["type"] == "image":
                            gemini_messages[-1]["parts"].append(
                                gemini_smart_process_image(message["content"][i]["image"])
                            )
                        else:
                            assert message["content"][i]["type"] == "text"
                            gemini_messages[-1]["parts"].append(glm.Part(text=message["content"][i]["text"]))
                else:
                    assert len(message["content"]) == 1 and message["content"][0]["type"] == "text"
                    gemini_messages[-1]["parts"].append(glm.Part(text=message["content"][0]["text"]))
            else:
                gemini_messages.append({
                    "role": "user",
                    "parts": [glm.Part(text=message["content"])],
                })
        else:
            gemini_messages.append({
                "role": "model",
                "parts": [glm.Part(text=message["content"])],
            })

    generated_text = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            gemini = genai.GenerativeModel(
                model_name=model,
                # generation_config=generation_config, # FIXME: temporary fix referring to https://github.com/google-gemini/generative-ai-python/issues/170#issuecomment-1891229158
                safety_settings=safety_settings
                )

            # convo = gemini.start_chat(history=[])
            
            # convo.send_message(gemini_messages)
            response = gemini.generate_content(gemini_messages)
            try:
                generated_text = response.text
                if response.candidates[0].finish_reason.name == "MAX_TOKENS":
                    generated_text += '...'
            except:
                for candidate in response.candidates:
                    generated_text = ' '.join([part.text for part in candidate.content.parts])
                    ic(">>> from candidate")
                    ic(generated_text)
                    break
            # output = convo.last.text
            break
        except genai.types.generation_types.StopCandidateException as e:
            print(type(e), e)
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    
    return generated_text


def chat_completion_cohere(model, messages, temperature, max_tokens):
    import cohere

    co = cohere.Client(os.environ["COHERE_API_KEY"])
    assert len(messages) > 0

    template_map = {"system":"SYSTEM",
                    "assistant":"CHATBOT",
                    "user":"USER"}

    assert messages[-1]["role"] == "user"
    prompt = messages[-1]["content"]

    if len(messages) > 1:
        history = []
        for message in messages[:-1]:
            history.append({"role":template_map[message["role"]], "message":message["content"]})
    else:
        history = None

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = co.chat(
                message=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                chat_history=history,
            )
            output = response.text
            break
        except cohere.core.api_error.ApiError as e:
            print(type(e), e)
            raise
        except Exception as e:
            print(type(e), e)
            break
    
    return output


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])
