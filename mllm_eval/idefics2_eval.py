"""pip install transformers>=4.35.2
"""
import os
import torch
import time
from typing import List
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from transformers.utils import is_flash_attn_2_available
import traceback


class Idefics2():
    support_multi_image = True
    merged_image_files = []
    def __init__(self, model_path:str="HuggingFaceM4/idefics2-8b", eval_mode=None) -> None:
        """Llava model wrapper

        Args:
            model_path (str): model name
        """
        attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else None
        print(f"Using {attn_implementation} for attention implementation")
        self.model = AutoModelForVision2Seq.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, _attn_implementation=attn_implementation).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
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

            text_prompt = temple_txt + inputs['above_content'] + '\n' + inputs['below_content'] + '\n' + temple_img
            messages = [
                {
                    "role": "user",
                    "content": [ {"type": "image"}] * len(inputs['images']) + [{"type": "text", "text": text_prompt}]
                }
            ]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            images = [load_image(image_link) for image_link in inputs['images']]
            inputs = self.processor(text=prompt, images=images, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            generate_ids = self.model.generate(**inputs, max_new_tokens=512, num_beams=1)
            generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return generated_text
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
            text_prompt = temple_txt + inputs['paragraphs'] + '\n' + temple_img
            messages = [
                {
                    "role": "user",
                    "content": [ {"type": "image"}] * 1 + [{"type": "text", "text": text_prompt}]
                }
            ]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            images = [load_image(inputs['image'])]
            inputs = self.processor(text=prompt, images=images, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            generate_ids = self.model.generate(**inputs, max_new_tokens=512, num_beams=1)
            generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return generated_text
        else:
            raise NotImplementedError

        
        
    def __del__(self):
        for image_file in self.merged_image_files:
            if os.path.exists(image_file):
                os.remove(image_file)
