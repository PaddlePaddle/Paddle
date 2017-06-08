#!/bin/env python

import paddle.cloud
# Init Paddle cloud with server addr and namespace
paddle.cloud.init(server="http://localhost:8080", \
                  namespace="paddle")

# Describe a bash job for prepare training data with bash script
job = paddle.cloud.job.BashJob(name="prepare-paddle-data", \
                           persistent_volume_claim_name="nfs-yanxu")

# Run the job in kubernetes cluster, specified trainner count is 3
job.run(filename="./get_data.sh", trainner_count=3)

# Waiting for the job complete
job.sync_wait(timeout=600)
