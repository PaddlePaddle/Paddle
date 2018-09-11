# 在OpenMPI集群中启动训练

## 准备OpenMPI集群

执行下面的命令以启动3个节点的OpenMPI集群和一个"head"节点：

```bash
paddle/scripts/cluster_train_v2/openmpi/docker_cluster
kubectl create -f head.yaml
kubectl create -f mpi-nodes.yaml
```

然后可以从head节点ssh无密码登录到OpenMPI的每个节点上。

## 启动集群作业

您可以按照下面的步骤在OpenMPI集群中提交paddle训练任务：

```bash
# 获得head和node节点的IP地址
kubectl get po -o wide
# 将node节点的IP地址保存到machines文件中
kubectl get po -o wide | grep nodes | awk '{print $6}' > machines
# 拷贝必要的文件到head节点
scp -i ssh/id_rsa.mpi.pub machines prepare.py train.py start_mpi_train.sh tutorial@[headIP]:~
# ssh 登录到head节点
ssh -i ssh/id_rsa.mpi.pub tutorial@[headIP]
# --------------- 以下操作均在head节点中执行 ---------------
# 准备训练数据
python prepare.py
# 拷贝训练程序和字典文件到每台MPI节点
cat machines | xargs -i scp word_dict.pickle train.py start_mpi_train.sh machines {}:/home/tutorial
# 创建日志目录
mpirun -hostfile machines -n 3 mkdir /home/tutorial/logs
# 拷贝训练数据到各自的节点
scp train.txt-00000 test.txt-00000 [node1IP]:/home/tutorial
scp train.txt-00001 test.txt-00001 [node2IP]:/home/tutorial
scp train.txt-00002 test.txt-00002 [node3IP]:/home/tutorial
# 启动训练任务
mpirun -hostfile machines -n 3  /home/tutorial/start_mpi_train.sh
```
