# PaddlePaddle Cluster

为了使用户不必学习Kubernetes中复杂的YAML配置以及各种概念，降低PaddlePaddle分布式训练的复杂程度，基于[Kubernetes Python Client](https://github.com/kubernetes-incubator/client-python) 提供Python版本的分布式训练API.

- 环境准备
通知系统管理在Kubernetes集群中根据不同的分布式存储类型创建PersistentVolume以及PersistentVolumeClaim，并得到创建的PersistentVolumeClaim,得到一个分布式存储的卷。


- 初始化集群配置

  ```python
  import paddle
  from paddle import PaddlePaddleCluster
  import time
  paddle_cluster = PaddlePaddleCluster(namespace="paddle", name="paddle-cluster-job")
  paddle_cluster.prepare_training_data.upload_local_file(filename="./get_data.sh", claim_name="nfs-k8s")
  ```

- 提交数据预处理任务,并等待其完成
Demo中通过一个[get_data.sh](./get_data.sh)脚本来下载数据并完成数据的切片操作

  ```python
  paddle_cluster.prepare_training_data.run(trainner_count=3)
  # Waiting for prepare data job complete
  while True:
      status = paddle_cluster.prepare_training_data.get_job_status()
      if status == paddle.JOB_STATUS_COMPLETE:
          print "Prepare training data complete"
          break
      print "Waiting for prepra training data for 10 seconds ..."
      time.sleep(10)

  ```

- 启动PaddlePaddle训练的Job
