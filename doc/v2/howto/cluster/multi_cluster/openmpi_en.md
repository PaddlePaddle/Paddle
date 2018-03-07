# OpenMPI

## Prepare an OpenMPI cluster

Run the following command to start a 3-node MPI cluster and one "head" node.

```bash
cd paddle/scripts/cluster_train_v2/openmpi/docker_cluster
kubectl create -f head.yaml
kubectl create -f mpi-nodes.yaml
```

Then you can log in to every OpenMPI node using ssh without input any passwords.

## Launching Cluster Job

Follow the steps to launch a PaddlePaddle training job in OpenMPI cluster:\

```bash
# find out node IP addresses
kubectl get po -o wide
# generate a "machines" file containing node IP addresses
kubectl get po -o wide | grep nodes | awk '{print $6}' > machines
# copy necessary files onto "head" node
scp -i ssh/id_rsa.mpi.pub machines prepare.py train.py start_mpi_train.sh tutorial@[headIP]:~
# login to head node using ssh
ssh -i ssh/id_rsa.mpi.pub tutorial@[headIP]
# --------------- in head node ---------------
# prepare training data
python prepare.py
# copy training data and dict file to MPI nodes
cat machines | xargs -i scp word_dict.pickle train.py start_mpi_train.sh machines {}:/home/tutorial
# creat a directory for storing log files
mpirun -hostfile machines -n 3 mkdir /home/tutorial/logs
# copy training data to every node
scp train.txt-00000 test.txt-00000 [node1IP]:/home/tutorial
scp train.txt-00001 test.txt-00001 [node2IP]:/home/tutorial
scp train.txt-00002 test.txt-00002 [node3IP]:/home/tutorial
# start the job
mpirun -hostfile machines -n 3  /home/tutorial/start_mpi_train.sh
```
