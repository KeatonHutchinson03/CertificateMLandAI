#!/bin/bash

accelerate launch session_3.py --experiment_name="CIFAR10" \
                        --path_to_data="./data" \
                        --working_directory="u/khutchinson/assignment3/" \
                        --epochs=50 \
                        --save_checkpoint_interval=10 \
                        --gradient_accumulation_steps=1 \
                        --batch_size=64 \
                        --learning_rate=0.001 \
                        --num_workers=4 \

### ANSWERS FOR Accelerate config
    #First question after downloading
        #This machine
        # MULTI-GPU
        #Could set yes or no for debugging purposes
        #NO
        #NO
        #NO
        #NO to all things that help accelerator
        #2
        #Which GPUs do you want to use?
            #Find the names when using ondemand
        #bf16 
        #config file saves and accelerate has everything it needs to get done
    