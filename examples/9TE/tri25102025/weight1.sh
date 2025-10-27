#!/bin/bash

for weight in 0.01 0.3 0.5; do
    CUDA_VISIBLE_DEVICES=1 python triclass_te_na_as_2_tno01.5.py --class_weight_na $weight
done
