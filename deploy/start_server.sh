#!/bin/bash
cd /media/ubuntu/Data/codes/jianxiong/memory-nav
export PATH="/home/ubuntu/miniconda3/envs/internvla/bin:$PATH"
export PYTHONPATH="/media/ubuntu/Data/codes/jianxiong/memory-nav:$PYTHONPATH"
exec /home/ubuntu/miniconda3/envs/internvla/bin/python deploy/ws_proxy_with_memory.py --gpu 1
