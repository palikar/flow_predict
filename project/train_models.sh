#!/bin/bash



python train.py --data ./data/generated_data/ --model-type 'c' --cuda --model-name 'res' --threads 4 --batch-size 32 --shuffle --epochs 100 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --g_layers 2 --g_nfg 16 --g_input_nc 6 --g_output_nc 6 --output_dir './plain_model_output_1'

python train.py --data ./data/generated_data/ --model-type 'c' --cuda --model-name 'res' --threads 4 --batch-size 32 --shuffle --epochs 100 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --g_layers 4 --g_nfg 16 --g_input_nc 6 --g_output_nc 6 --output_dir './plain_model_output_2'

python train.py --data ./data/generated_data/ --model-type 'c' --cuda --model-name 'res' --threads 4 --batch-size 32 --shuffle --epochs 100 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --g_layers 6 --g_nfg 16 --g_input_nc 6 --g_output_nc 6 --output_dir './plain_model_output_3'



python train.py --data ./data/generated_data/ --model-type 'c' --cuda --model-name 'res' --threads 4 --batch-size 32 --shuffle --epochs 100 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --g_layers 2 --g_nfg 32 --g_input_nc 6 --g_output_nc 6 --output_dir './plain_model_output_4'

python train.py --data ./data/generated_data/ --model-type 'c' --cuda --model-name 'res' --threads 4 --batch-size 32 --shuffle --epochs 100 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --g_layers 4 --g_nfg 32 --g_input_nc 6 --g_output_nc 6 --output_dir './plain_model_output_5'

python train.py --data ./data/generated_data/ --model-type 'c' --cuda --model-name 'res' --threads 4 --batch-size 32 --shuffle --epochs 100 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --g_layers 6 --g_nfg 32 --g_input_nc 6 --g_output_nc 6 --output_dir './plain_model_output_6'
