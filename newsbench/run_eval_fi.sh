model_name_list=(
    'llava' # 0
    'llava' # 1
    'qwenVL' # 2
    'minicpm-V-2.5' # 3
    'idefics2' # 4
    'instructblip' # 5
    'fuyu' # 6
    'kosmos2' # 7
)

model_path_list=(
    '/PATH_TO_YOURS/models/llava-hf/llava-1.5-7b-hf'
    '/PATH_TO_YOURS/models/llava-hf/llava-1.5-13b-hf'
    '/PATH_TO_YOURS/models/Qwen/Qwen-VL-Chat'
    '/PATH_TO_YOURS/models/openbmb/MiniCPM-Llama3-V-2_5'
    '/PATH_TO_YOURS/models/HuggingFaceM4/idefics2-8b'
    '/PATH_TO_YOURS/models/Salesforce/instructblip-flan-t5-xxl'
    '/PATH_TO_YOURS/models/adept/fuyu-8b'
    '/PATH_TO_YOURS/models/microsoft/kosmos-2-patch14-224'
)

eval_mode_list=(
    'flow_insert_1_cn' 
    'flow_insert_2_cn'
    'flow_insert_3_cn'
    'flow_insert_1_en'
    'flow_insert_2_en'
    'flow_insert_3_en'
)

endfix=json

for eval_mode in "${eval_mode_list[@]}"; do
    dataset_path=/PATH_TO_YOURS_ftii_data/newsinsertbench_${eval_mode}.${endfix}

    # Select the image path based on eval mode
    if [[ "${eval_mode}" == *"_cn" ]]; then
        img_path=/PATH_TO_YOURS/images/NewsImages_cn_jpg
    else
        img_path=/PATH_TO_YOURS/images
    fi

    for i in {0..7}; do
        export CUDA_VISIBLE_DEVICES=$i

        python eval_mllm.py \
            --model_name "${model_name_list[$i]}" \
            --model_path "${model_path_list[$i]}" \
            --dataset_path ${dataset_path} \
            --img_path ${img_path} \
            --results_dir /PATH_TO_YOURS/news_bench/FinalBench/results \
            --eval_mode ${eval_mode} &
    done

    wait
done
