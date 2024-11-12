"""pip install transformers>=4.35.2 transformers_stream_generator torchvision tiktoken chardet matplotlib
""" 
import tempfile
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from typing import List
from utils import merge_images, load_image
import re
import traceback

class QwenVL():
    support_multi_image = False
    merged_image_files = []
    def __init__(self, model_path:str="Qwen/Qwen-VL-Chat", eval_mode=None) -> None:
        """
        Args:
            model_path (str): Qwen model name, e.g. "Qwen/Qwen-VL-Chat"
        """
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, bf16=True).eval()
        self.eval_mode = eval_mode
    
    def __call__(self, inputs: dict) -> str:
        if self.eval_mode.startswith('single_choice'):
            try:
                generated_text = self.get_single_choice_anwser(inputs)
            except:
                return 'ERROR!!!'
        elif self.eval_mode.startswith('flow_insert'):
            try:
                generated_text = self.get_flow_insert_answer(inputs)
            except:
                return 'ERROR!!!'
        else:
            raise NotImplementedError
        return generated_text
        
    def get_single_choice_anwser(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'above_content':
                'below_content: 
                'images': [
                    
                ]
                'temple_img': 
                'temple_txt': 
            }
        """
        temple_txt = inputs['temple_txt']
        temple_img = inputs['temple_img']
        if self.support_multi_image:
            raise NotImplementedError
        else:
            text_prompt = temple_txt + inputs['above_content'] + '\n' + inputs['below_content']
            text_prompt = text_prompt + temple_img
            inputs = self.prepare_prompt(inputs['images'], text_prompt)
            # Generate
            generated_text = self.get_parsed_output(inputs)
            return generated_text
        
    def get_flow_insert_answer(self, inputs: dict) -> str:
        """
        Args:
            inputs : {
                'paragraphs': 
                'image': 
                'temple_img': 
                'temple_txt': 
            }
        """
        temple_txt = inputs['temple_txt']
        temple_img = inputs['temple_img']
        if self.support_multi_image:
            raise NotImplementedError
        else:
            text_prompt = temple_txt + inputs['paragraphs']
            text_prompt = text_prompt + "\n<image>\n" + temple_img
            inputs = self.prepare_prompt(inputs['image'], text_prompt)
            # Generate
            generated_text = self.get_parsed_output(inputs)
            return generated_text
        

    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        if type(image_links) == str:
            image_links = [image_links]
        true_image_links = []
        for i, image_link in enumerate(image_links):
            if isinstance(image_link, str):
                true_image_links.append(image_link)
            elif isinstance(image_link, Image.Image):
                image_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                image_file.close()
                image_link.save(image_file.name)
                self.merged_image_files.append(image_file.name)
                true_image_links.append(image_file.name)
            else:
                raise NotImplementedError
        image_links = true_image_links
        input_list = []
        for i, image_link in enumerate(image_links):
            input_list.append({'image': image_link})
        input_list.append({'text': text_prompt})
        query = self.tokenizer.from_list_format(input_list)
        return query
    

    def get_parsed_output(self, query):
        response, history = self.model.chat(self.tokenizer, query=query, history=None)
        return response

    def __del__(self):
        for image_file in self.merged_image_files:
            os.remove(image_file)