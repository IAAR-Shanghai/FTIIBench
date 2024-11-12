# FTIIBench
**(ARXIV24)** This is the official code repository for *"FTII-Bench: A Comprehensive Multimodal Benchmark for Flow Text with Image Insertion."*

## Dataset
Our FTII-Bench of CN and EN type could be found in ./ftii_data/

And the images of FTII-Bench could be download from [Google Drive](https://drive.google.com/file/d/1eDGu-zsDYDvK_Y_mUmYV-npyceT_ur8f/view?usp=sharing)

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

