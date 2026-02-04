#!/bin/bash
cd /media/ubuntu/Data/codes/jianxiong/MemoryNav
export PATH="/home/ubuntu/miniconda3/envs/internvla/bin:$PATH"
export PYTHONPATH="/media/ubuntu/Data/codes/jianxiong/MemoryNav:$PYTHONPATH"
exec /home/ubuntu/miniconda3/envs/internvla/bin/python deploy/ws_proxy_with_memory.py --gpu 1
