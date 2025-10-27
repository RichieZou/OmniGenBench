#!/bin/bash

for weight in 0.01 0.3 0.5 0.7; do
    CUDA_VISIBLE_DEVICES=0 python triclass_te_na_as_2_tno0.py --class_weight_na $weight
done
