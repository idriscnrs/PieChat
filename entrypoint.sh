#!/bin/sh

set -x

# Run the first command
python3 -m piechat --config_file ./config_docker.ini --make_vdb

# Run the second command
python3 -m piechat --config_file ./config_docker.ini

# Create the vector database
# RUN NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     python3 -m piechat --config_file ./config_sofi.ini \
#     --make_vdb --data_path ./idris_doc_dataset

# Run the application
# RUN NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     python3 -m piechat --config_file ./config_sofi.ini --run_server
