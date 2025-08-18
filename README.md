# FTIIBench
**(ARXIV24)** This is the official code repository for *"FTII-Bench: A Comprehensive Multimodal Benchmark for Flow Text with Image Insertion."*

## Dataset

The text and images of FTII-Bench could be download from [Baidu Drive](https://pan.baidu.com/s/1ujROnxZCHi3ezz2-Etcvyg?pwd=wn7a)

### **Note that the data is only used for research purposes!**

## Evaluation
1. Set the appropriate paths in the run_eval_fi and run_eval_sc scripts.   
   ```bash
   bash run_eval_fi.sh # for flow insertion tasks
   bash run_eval_sc.sh # for single choice tasks
   ```
2. For evaluating with BGE models You can run ./mllm_eval/bge_eval.ipynb in the Jupyter environment.  

## Acknowledgement
Thanks to the open-source code from [Mantis](https://tiger-ai-lab.github.io/Mantis/)

