#!/bin/bash

# from workstation
ssh dumbo -X
# LOG into the folder
# cd /scratch/sri223/segments/final/

# Load necessary modules
module purge
module load java/1.8.0_72
module load spark/2.2.0
module load anaconda3/4.3.1
source activate mypy3


pyspark

# module load firefox/45.8.0
# firefox http://localhost:8888/tree?token=458fd59e91c9ca0354a45f6e70d1050ae8c968406e3082bf


# for i in /scratch/sri223/segments/final/*.pickle;
#     do echo "${i##*/}";
# done
