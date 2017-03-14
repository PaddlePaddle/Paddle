# PaddlePaddle Cluster

为了使用户不必学习Kubernetes中复杂的YAML配置以及各种概念，降低PaddlePaddle分布式训练的复杂程度，基于[Kubernetes Python Client](https://github.com/kubernetes-incubator/client-python) 提供Python版本的分布式训练API.

- 环境准备
通知系统管理在Kubernetes集群中根据不同的分布式存储类型创建PersistentVolume以及PersistentVolumeClaim，并得到创建的PersistentVolumeClaim,得到一个分布式存储的卷。


- 初始化PaddleCloud的配置，指定namespace

```python
import paddle.cloud
paddle.cloud.init(server=<your k8s server>,namespace=<your namespace>)
```
- 创建一个处理Bash脚本的Job,并提交到集群中运行,需要指定**Job**名以及管理员分配的**persistent volume claim**

```python
job = paddle.cloud.BashJob(name=<job name>, \
  persistent_volume_claim_name=<persistent volume claim>)
job.run()
```

- 同步的等待这个Job执行完成

```python
job.sync_wait(timeout=100)
```

- 创建一个PaddlePaddle的训练Job，
