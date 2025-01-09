#!/bin/bash

#$ -pe smp 1        # Specify parallel environment and legal core size
#$ -q gpu 
#$ -l gpu_card=1
#$ -N sam2_webui  # Specify job name

module load conda
conda activate sam2

fsync /afs/crc.nd.edu/user/j/jmangion/Public/sam2-video-webui/gradio_link.txt &
python3 test.py > gradio_link.txt
#python3 test.py

#stdbuf -oL python3 test.py > gradio_link.txt
