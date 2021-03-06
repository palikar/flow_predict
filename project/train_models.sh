#!/bin/bash


NGF=32
LAYERS=5
MODEL_CNT=${MODEL_CNT:-10}
NETS=()

MASK=${MASK:-"--no-mask"}
CUDA=${CUDA:-"--cuda"}

EPOCHS=${EPOCHS:-30}


while [ "$1" != "" ]; do
    
    case $1 in
        
        --pressure | -p)
            PRESSURE=1
            ;;
        --np-pressure | -np)
            NO_PRESSURE=1
            ;;
	
        --unet | -u)
            NETS=( "unet" "${NETS[@]}" )
            ;;
	
        --res | -r)
            NETS=( "res" "${NETS[@]}" )
            ;;
	
        --constant | -c)
            CONST=1
            ;;				
	
        --speed | -s)
            SPEED=1
            ;;
	
        --fluid | -f)
            FLUID=1
            ;;

        --layers | -l)
            LAYERS=`echo $2`
            shift
            ;;

        --ngf | -n)
            NGF=`echo $2`
            shift
            ;;

	--model-cnt | -m)
	    MODEL_CNT=`echo $2`
	    shift
	    ;;
	
	
	*)
	    echo "ERROR: unknown parameter \"$1\""
	    exit 1
	    ;;
    esac
    shift 1
    
    
done


echo "MASK: " $MASK
echo "CUDA: " $CUDA
echo "MODEL_CNT: " $MODEL_CNT
echo "NETS: " ${NETS[@]}
echo "EPOCHS: " ${EPOCHS}



# some_command &
# P1=$!
# other_command &
# P2=$!
# wait $P1 $P2

for net in ${NETS[@]}
do
    ##########################################################
    #  ____  _       _         __  __           _      _     #
    # |  _ \| | __ _(_)_ __   |  \/  | ___   __| | ___| |___ #
    # | |_) | |/ _` | | '_ \  | |\/| |/ _ \ / _` |/ _ \ / __|#
    # |  __/| | (_| | | | | | | |  | | (_) | (_| |  __/ \___ #
    # |_|   |_|\__,_|_|_| |_| |_|  |_|\___/ \__,_|\___|_|___/#
    ##########################################################

    if [ ! -z "$CONST" ]; then

	if [ ! -z "$PRESSURE" ]; then
            for i in $(seq 1 ${MODEL_CNT}); do
                
                python train.py --data ./data/generated_data/ --model-type 'c' ${CUDA} --model-name "${net}" --threads 4 --batch-size 3 --shuffle --epochs ${EPOCHS} --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir "./results_c/plain_results_$i_${RANDOM}/" --evaluate --g_nfg ${NGF} --g_layers ${LAYERS} --use-pressure  ${MASK}

	    done
        fi


        if [ ! -z "$NO_PRESSURE" ]; then
            for i in $(seq 1 ${MODEL_CNT}); do
                
                python train.py --data ./data/generated_data/ --model-type 'c' ${CUDA} --model-name "${net}" --threads 4 --batch-size 3 --shuffle --epochs ${EPOCHS} --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir "./results_c/plain_results_$i_${RANDOM}/" --evaluate --g_nfg ${NGF} --g_layers ${LAYERS} ${MASK}
                
	    done
        fi

        
	

    fi

    #######################################################
    #######################################################
    #######################################################


    #################################################################
    #  ____                      _   __  __           _      _      #
    # / ___| _ __   ___  ___  __| | |  \/  | ___   __| | ___| |___  #
    # \___ \| '_ \ / _ \/ _ \/ _` | | |\/| |/ _ \ / _` |/ _ \ / __| #
    #  ___) | |_) |  __/  __/ (_| | | |  | | (_) | (_| |  __/ \__   #
    # |____/| .__/ \___|\___|\__,_| |_|  |_|\___/ \__,_|\___|_|___/ #
    #       |_|                                                     #
    #################################################################
    
    if [ ! -z "$SPEED" ]; then

	if [ ! -z "$PRESSURE" ]; then
            for i in $(seq 1 ${MODEL_CNT}); do

		python train.py --data ./data/generated_data/ --model-type 's' ${CUDA} --model-name "${net}" --threads 4 --batch-size 3 --shuffle --epochs ${EPOCHS} --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir "./results_s/plain_results_${i}_${RANDOM}/" --evaluate --g_nfg ${NGF} --g_layers ${LAYERS} --use-pressure ${MASK}

	    done
        fi

        if [ ! -z "$NO_PRESSURE" ]; then
	    for i in $(seq 1 ${MODEL_CNT}); do

	        python train.py --data ./data/generated_data/ --model-type 's' ${CUDA} --model-name "${net}" --threads 4 --batch-size 3 --shuffle --epochs ${EPOCHS} --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir "./results_s/plain_results_${i}_${RANDOM}/" --evaluate --g_nfg ${NGF} --g_layers ${LAYERS} ${MASK}

	    done
        fi
    fi

    #################################################################
    #################################################################
    #################################################################

    #################################################################
    #  _____ _       _     _   __  __           _      _            #
    # |  ___| |_   _(_) __| | |  \/  | ___   __| | ___| |___        #
    # | |_  | | | | | |/ _` | | |\/| |/ _ \ / _` |/ _ \ / __|       #
    # |  _| | | |_| | | (_| | | |  | | (_) | (_| |  __/ \__         #
    # |_|   |_|\__,_|_|\__,_| |_|  |_|\___/ \__,_|\___|_|___/       #
    #################################################################


    if [ ! -z "$FLUID" ]; then
	if [ ! -z "$PRESSURE" ]; then

	    for i in $(seq 1 ${MODEL_CNT}); do

		python train.py --data ./data/generated_data/ --model-type 'vd' ${CUDA} --model-name "${net}" --threads 4 --batch-size 3 --shuffle --epochs ${EPOCHS} --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir "./results_vd/plain_results_$i_${RANDOM}/" --evaluate --g_nfg ${NGF} --g_layers ${LAYERS} --use-pressure ${MASK}

	    done
	fi

        if [ ! -z "$NO_PRESSURE" ]; then
	    for i in $(seq 1 ${MODEL_CNT}); do

	        python train.py --data ./data/generated_data/ --model-type 'vd' ${CUDA} --model-name "${net}" --threads 4 --batch-size 3 --shuffle --epochs ${EPOCHS} --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir "./results_vd/plain_results_$i_${RANDOM}/" --evaluate --g_nfg ${NGF} --g_layers ${LAYERS}  ${MASK}

	    done
        fi

    fi

    #################################################################
    #################################################################
    #################################################################
    
    
    
done



python train.py --data ./data/generated_data/ --model-type 'c' --model-name "time_1" --threads 4 --batch-size 3 --shuffle --epochs 1 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir "./results_c/time_${RANDOM}/" --evaluate --g_nfg 32 --g_layers 5 --use-pressure --no-train
