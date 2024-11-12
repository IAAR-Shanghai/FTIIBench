import json
import regex as re
import os
import datasets
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import sys
from PIL import Image
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.join(os.getcwd(), '..')))
from mllm_eval import MLLM_Models
from utils import load
from typing import List
import re
import pandas as pd
import random
import torch
import argparse


def eval_single_choice(benchmark_data, benchmark_root_path, model, eval_mode):
    def generate_instruction_string(img_list_len, eval_mode):
        labels = ['A', 'B', 'C', 'D']
        chosen_labels = labels[:img_list_len]
        labels_str = ', '.join(chosen_labels)

        if eval_mode.split('_')[-1] == 'en':
            return (f"Given {img_list_len} images, labeled sequentially as {labels_str}. Please select the image most suitable to be inserted after the given news paragraph. You only need to provide the label of the image.",
                    f"Given the following news paragraph: ")
        elif eval_mode.split('_')[-1] == 'cn':
            return (f"给定的{img_list_len}张图片，依次被标记为{labels_str}。请从中选出最适合插入到给定新闻文本中的图片，你仅需要回答图片标签即可。",
                    f"给定以下新闻段落：")
        else: raise ValueError

    def find_answer_in_string(text):
        # Define the regular expression pattern to match letters A, B, C, D
        # This pattern ensures the letter is not part of a larger word or phrase
        pattern = r'\b[A-D]\b'
        
        # Search for the pattern in the text
        match = re.search(pattern, text.upper())
        
        if match:
            return match.group(0)
        
        # Additional pattern to handle cases where the letter is not surrounded by word boundaries
        pattern = r'[A-D]'
        match = re.search(pattern, text.upper())
        
        if match:
            return match.group(0)
        
        return None
        
    correct = 0
    results = []

    for _, row in tqdm(benchmark_data.iterrows(), desc=f'Evaluating', ncols=100):
        images_list = []
        # Iterate over each column in the row
        for column_name in benchmark_data.columns:
            # Check if the column name starts with 'img'
            if column_name.startswith('img') and not column_name.startswith('url') and not pd.isna(row[column_name]):
                images_list.append(benchmark_root_path + '/' + row[column_name])
        
        # Generate the instruction string
        temple_img, temple_txt = generate_instruction_string(len(images_list), eval_mode)

        if pd.isna(row['below_content']): row['below_content'] = ''
        if pd.isna(row['above_content']): row['above_content'] = ''
        inputs = {
            'above_content': row['above_content'],
            'below_content': row['below_content'],
            'images': images_list,
            'temple_img': temple_img,
            'temple_txt': temple_txt,
        }
        
        # Get the model's raw answer
        raw_answer = model(inputs)
        model_answer = find_answer_in_string(raw_answer)
        true_label = row['Answer']
        
        # Check if the model's answer is correct
        if model_answer == true_label:
            correct += 1
        
        # Append the results to the list
        result_row = row.to_dict()
        result_row['raw_answer'] = raw_answer
        result_row['model_answer'] = model_answer
        results.append(result_row)

    # Calculate the accuracy
    accuracy = correct / len(benchmark_data) #############################

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    # Convert the DataFrame to JSON format
    results_json = results_df.to_dict(orient='records')
    
    # Combine accuracy and results into one dictionary
    output = {
        'accuracy': accuracy,
        'results': results_json
    }
    
    return output



from tqdm import tqdm
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def eval_flow_insert(benchmark_data, benchmark_root_path, model, eval_mode):

    def generate_instruction_string(eval_mode):
        if eval_mode.split('_')[-1] == 'en':
            return (f"Do you think the content of this image is suitable as an illustration for the given text paragraph? If it is not suitable, respond with 'NO'; if it is suitable, respond with a suitability score between 0 and 100.",
                    f"Given the following paragraphs: ")
        elif eval_mode.split('_')[-1] == 'cn':
            return ("你认为这张图片的内容适合作为给定文本段落的配图吗？如果不适合，仅回答否；如果适合，仅回答一个适配程度的值，在0到100之间。", #如果适合，请仅回答一个匹配程度值，在0到100之间。否则，仅返回'NO'。
                    f"给定以下段落：")
        else: raise ValueError

    def find_confidence_in_string(text):
        # Define the regular expression pattern to match numbers between 0 and 100
        pattern = r'(?:100(?:\.0{1,})?|(?:[1-9]?\d|0)(?:\.\d{1,})?)'
        
        # Search for the pattern in the text
        match = re.search(pattern, text)
        
        if match:
            return float(match.group())
        else:
            return None
        
    # Initialize counters for each evaluation condition
    correct_matches = 0
    total_matches = 0
    correct_image_matches = 0
    total_image_matches = 0
    correct_blank_matches = 0
    total_blank_matches = 0

    all_true_labels = []
    all_model_answers = []
    news_item_results = []
    temple_img, temple_txt = generate_instruction_string(eval_mode)

    for content in tqdm(benchmark_data):
        image_database = content['imagedatabase']

        for news_item in tqdm(content['news_text'], desc="News Items"):
            selected_images = set()
            paragraph_content = news_item['content']
            model_answer = ["" for i in range(len(paragraph_content))]
            true_label = news_item['groundtruth']
            inputs_paragraphs = ''
            news_id = news_item['id']

            for index_paragraph, paragraph in enumerate(paragraph_content):
                inputs_paragraphs += '\n' + paragraph
                highest_confidence = -float('inf')
                best_image_idx = None
                
                for index_img, img in enumerate(image_database):
                    if index_img in selected_images:
                        continue
                    inputs = {
                        'paragraphs': inputs_paragraphs,
                        'image': benchmark_root_path + '/' + img[0],
                        'temple_img': temple_img,
                        'temple_txt': temple_txt,
                    }

                    # Get the model's raw answer
                    raw_answer = model(inputs)
                    confidence = find_confidence_in_string(raw_answer)
                    print(raw_answer, confidence)
                    
                    if confidence is not None:
                        if confidence > highest_confidence:
                            highest_confidence = confidence
                            best_image_idx = index_img

                if best_image_idx is not None:
                    selected_images.add(best_image_idx)
                    model_answer[index_paragraph] = image_database[best_image_idx][0]
            
            all_true_labels.extend(true_label)
            all_model_answers.extend(model_answer)

            for model_answer_item, true_label_item in zip(model_answer, true_label):
                # Calculate for all matches
                if model_answer_item == true_label_item:
                    correct_matches += 1
                total_matches += 1

                if true_label_item != "":
                    if model_answer_item == true_label_item:
                        correct_image_matches += 1
                    total_image_matches += 1

                if true_label_item == "":
                    if model_answer_item == true_label_item:
                        correct_blank_matches += 1
                    total_blank_matches += 1

            news_item_results.append({
                'id': news_id,
                'true_label': true_label,
                'model_answer': model_answer
            })

    performance_metrics = {
        'overall_accuracy': correct_matches / total_matches if total_matches > 0 else 0,
        'image_only_accuracy': correct_image_matches / total_image_matches if total_image_matches > 0 else 0,
        'blank_only_accuracy': correct_blank_matches / total_blank_matches if total_blank_matches > 0 else 0
    }

    return {
        'performance_metrics': performance_metrics,
        'news_item_results': news_item_results,
    }





def main(
    model_name: str,
    model_path: str,
    dataset_path: str,
    results_dir: str,
    img_path = None,
    eval_mode: str='single_choice',
    seed = 42
):
    random.seed(seed)
    benchmark_data = load(dataset_path)
    benchmark_root_path = os.path.dirname(os.path.abspath(dataset_path)) if img_path == 'None' else img_path

    if model_name == "random":
        model = None
    else:
        model = MLLM_Models(model_name, model_path, eval_mode)

    os.makedirs(results_dir, exist_ok=True)

    if eval_mode.startswith('single_choice'):
        results_dict = eval_single_choice(benchmark_data, benchmark_root_path, model, eval_mode)
        
        json_path = os.path.join(results_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=4)
        
        del model
        torch.cuda.empty_cache()
        return results_dict
    
    elif eval_mode.startswith('flow_insert'):
        results_dict = eval_flow_insert(benchmark_data, benchmark_root_path, model, eval_mode)
        
        json_path = os.path.join(results_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=4)

        del model
        torch.cuda.empty_cache()
        return results_dict
    else:
        raise NotImplementedError
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate volume of a cylinder')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--img_path', type=str, default='None')
    parser.add_argument('--results_dir', type=str, default='None')
    parser.add_argument('--eval_mode', type=str, default='single_choice_1')
    args = parser.parse_args()

    results_dir = args.results_dir + '/' + args.eval_mode + '/' + args.model_path.split('/')[-1]

    results = main(
        model_name=args.model_name,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        results_dir=results_dir,
        img_path=args.img_path,
        eval_mode=args.eval_mode,
        seed=42
    )