nohup spark-submit recommendation2.py --conf spark.yarn.submit.waitAppCompletion=false  --driver-memory 16g --executor-memory 16g --num-executors 4 --executor-cores 4  --deploy-mode cluster > spark-submit.log 2>&1 &

#86917
