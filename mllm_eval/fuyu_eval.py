"""need latest transformers from source
pip install transformers>=4.35.2
"""
import requests
import torch
from transformers import FuyuProcessor, FuyuForCausalLM, AutoTokenizer
from PIL import Image
from typing import List
from io import BytesIO
from utils import merge_images, load_image
import re
import traceback

class Fuyu():
    support_multi_image = False
    def __init__(self, model_path:str="adept/fuyu-8b", eval_mode=None) -> None:
        """
        Args:
            model_path (str): Fuyu model name, e.g. "adept/fuyu-8b"
        """
        self.model_path = model_path
        self.processor = FuyuProcessor.from_pretrained(model_path)
        self.model = FuyuForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
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
            text_prompt = temple_txt + inputs['above_content'] + '\n' + inputs['below_content'] + '\n' + temple_img
            inputs = self.prepare_prompt(inputs['images'], text_prompt)
            return self.get_parsed_output(inputs)


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
            text_prompt = temple_txt + inputs['paragraphs'] + '\n' + temple_img
            inputs = self.prepare_prompt(inputs['image'], text_prompt)
            return self.get_parsed_output(inputs)


    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        if type(image_links) == str:
            image_links = [image_links]
        image = merge_images(image_links)
        inputs = self.processor(text=text_prompt, images=image, return_tensors="pt").to(self.model.device)
        return inputs
    
    def get_parsed_output(self, inputs):
        generation_output = self.model.generate(**inputs, max_new_tokens=512, pad_token_id=self.pad_token_id)
        input_len = inputs.input_ids.shape[1]
        generation_text = self.processor.batch_decode(generation_output[:, input_len:], skip_special_tokens=True)
        return generation_text[0].strip(" \n")
