"""pip install transformers>=4.35.2
"""
import os
import torch
import time
from PIL import Image
from typing import List
from transformers import AutoModel, AutoTokenizer
from transformers.image_utils import load_image
from transformers.utils import is_flash_attn_2_available
import re
import traceback

class MiniCPMV():
    support_multi_image = True
    def __init__(self, model_path:str="openbmb/MiniCPM-Llama3-V-2_5", eval_mode=None) -> None:
        """Llava model wrapper

        Args:
            model_path (str): model name
        """
        
        attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else None
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map='cuda', _attn_implementation=attn_implementation).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.eval_mode = eval_mode

        print(f"Using {attn_implementation} for attention implementation")

        
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
            content = []
            text_prompt = temple_txt + inputs['above_content'] + '\n' + inputs['below_content']
            content.append(text_prompt)
            for img in inputs["images"]:
                if isinstance(img, str): img = load_image(img)
                elif isinstance(img, Image.Image): pass
                else: raise ValueError("Invalid image input", img, "should be str or PIL.Image.Image")
                content.append(img)
            content.append(temple_img)
            messages = [{"role": "user", "content": content}]
            res = self.model.chat(
                image=None,
                msgs=messages,
                tokenizer=self.tokenizer,
                sampling=False, # if sampling=False, beam_search will be used by default
            )
            return res
        else:
            raise NotImplementedError
        
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
            content = []
            text_prompt = temple_txt + inputs['paragraphs']
            content.append(text_prompt)
            if isinstance(inputs["image"], str): img = load_image(inputs["image"])
            elif isinstance(inputs["image"], Image.Image): img = inputs["image"]
            else: raise ValueError("Invalid image input", inputs["image"], "should be str or PIL.Image.Image")
            content.append(img)
            content.append(temple_img)
            messages = [{"role": "user", "content": content}]
            res = self.model.chat(
                image=None,
                msgs=messages,
                tokenizer=self.tokenizer,
                sampling=False, # if sampling=False, beam_search will be used by default
            )
            return res
        else:
            raise NotImplementedError
        


    