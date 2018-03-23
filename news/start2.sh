#!/bin/bash
module load anaconda3/4.3.1
module load spark/2.2.0
nohup spark-submit classifier2.py  > spark-submit.log 2>&1 & 

