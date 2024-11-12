"""pip install accelerate transformers>=4.35.2
BLIP_FLANT5 tends to otuput shorter text, like "a tiger and a zebra". Try to design the prompt with shorter answer.
"""
import requests
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from typing import List
import torch
from typing import List
from io import BytesIO
from utils import merge_images, load_image
import traceback

class INSTRUCTBLIP_FLANT5():
    support_multi_image = False
    def __init__(self, model_path:str="Salesforce/instructblip-flan-t5-xxl", eval_mode=None) -> None:
        """
        Args:
            model_path (str): BLIP_FLANT5 model name, e.g. "Salesforce/blip2-flan-t5-xxl"
        """
        self.model_path = model_path
        self.processor = InstructBlipProcessor.from_pretrained(model_path)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
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
        inputs = self.processor(image, text_prompt, return_tensors="pt").to(self.model.device)
        return inputs

    def get_parsed_output(self, inputs):
        generation_output = self.model.generate(**inputs, 
            do_sample=True,
            # num_beams=5,
            max_new_tokens=512,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )
        generation_text = self.processor.batch_decode(generation_output, skip_special_tokens=True)
        return generation_text[0].strip(" \n")
 