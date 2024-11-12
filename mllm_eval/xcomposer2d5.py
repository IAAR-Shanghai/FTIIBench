"""pip install transformers>=4.35.2
"""
import torch
from transformers import AutoModel, AutoTokenizer

class XComposer2d5():
    support_multi_image = True
    def __init__(self, model_path:str="internlm/internlm-xcomposer2d5-7b", eval_mode=None) -> None:

        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model.tokenizer = self.tokenizer
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
            query = temple_txt + inputs['above_content'] + '\n' + inputs['below_content'] \
                + temple_img + 'A: <ImageHere>; B: <ImageHere>; C: <ImageHere>'
            image = inputs['images']
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                generated_text, _ = self.model.chat(self.tokenizer, query, image, do_sample=False, num_beams=1, use_meta=True)

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
            query = temple_txt + inputs['paragraphs'] + '\n' \
                + temple_img + '<ImageHere>'
            image = [inputs['image']]
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                generated_text, _ = self.model.chat(self.tokenizer, query, image, do_sample=False, num_beams=1, use_meta=True)

            return generated_text
        else:
            raise NotImplementedError