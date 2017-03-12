#!/bin/env python
#-*-coding:utf-8-*-
import paddle
from paddle import PaddlePaddleCluster
import time
paddle_cluster = PaddlePaddleCluster(namespace="yancey", name="paddle-cluster-job")
paddle_cluster.prepare_training_data.upload_local_file("./get_data.sh", "nfs-k8s")
paddle_cluster.prepare_training_data.run(trainner_count=3)

# Waiting for prepare data job complete
while True:
    status = paddle_cluster.prepare_training_data.get_job_status()
    if status == paddle.JOB_STATUS_COMPLETE:
        print "Prepare training data complete"
        break
    print "Waiting for prepra training data for 10 seconds ..."
    time.sleep(10)
