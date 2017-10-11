# Kubernetes 简介

[*Kubernetes*](http://kubernetes.io/)是Google开源的容器集群管理系统，其提供应用部署、维护、扩展机制等功能，利用Kubernetes能方便地管理跨机器运行容器化的应用。Kubernetes可以在物理机或虚拟机上运行，且支持部署到[AWS](http://kubernetes.io/docs/getting-started-guides/aws)，[Azure](http://kubernetes.io/docs/getting-started-guides/azure/)，[GCE](http://kubernetes.io/docs/getting-started-guides/gce)等多种公有云环境。介绍分布式训练之前，需要对[Kubernetes](http://kubernetes.io/)有一个基本的认识，下面先简要介绍一下本文用到的几个Kubernetes概念。

- [*Node*](http://kubernetes.io/docs/admin/node/) 表示一个Kubernetes集群中的一个工作节点，这个节点可以是物理机或者虚拟机，Kubernetes集群就是由node节点与master节点组成的。

- [*Pod*](http://kubernetes.io/docs/user-guide/pods/) 是一组(一个或多个)容器，pod是Kubernetes的最小调度单元，一个pod中的所有容器会被调度到同一个node上。Pod中的容器共享NET，PID，IPC，UTS等Linux namespace。由于容器之间共享NET namespace，所以它们使用同一个IP地址，可以通过*localhost*互相通信。不同pod之间可以通过IP地址访问。

- [*Job*](http://kubernetes.io/docs/user-guide/jobs/) 描述Kubernetes上运行的作业，一次作业称为一个job，通常每个job包括一个或者多个pods，job启动后会创建这些pod并开始执行一个程序，等待这个程序执行成功并返回0则成功退出，如果执行失败，也可以配置不同的重试机制。

- [*Volume*](http://kubernetes.io/docs/user-guide/volumes/) 存储卷，是pod内的容器都可以访问的共享目录，也是容器与node之间共享文件的方式，因为容器内的文件都是暂时存在的，当容器因为各种原因被销毁时，其内部的文件也会随之消失。通过volume，就可以将这些文件持久化存储。Kubernetes支持多种volume，例如hostPath(宿主机目录)，gcePersistentDisk，awsElasticBlockStore等。

- [*Namespaces*](https://kubernetes.io/docs/user-guide/namespaces/) 命名空间，在kubernetes中创建的所有资源对象(例如上文的pod，job)等都属于一个命名空间，在同一个命名空间中，资源对象的名字是唯一的，不同空间的资源名可以重复，命名空间主要为了对象进行逻辑上的分组便于管理。本文只使用了默认命名空间。

- [*PersistentVolume*](https://kubernetes.io/docs/user-guide/persistent-volumes/): 和[*PersistentVolumeClaim*](https://kubernetes.io/docs/user-guide/persistent-volumes/#persistentvolumeclaims)结合，将外部的存储服务在Kubernetes中描述成为统一的资源形式，便于存储资源管理和Pod引用。

## 部署Kubernetes集群

Kubernetes提供了多种集群部署的方案，本文档内不重复介绍。这里给出集中常见的部署方法：

- [*minikube*](https://kubernetes.io/docs/getting-started-guides/minikube/): 快速在本地启动一个单机的kubernetes服务器，便于本地验证和测试。
- [*kubeadm*](http://kubernetes.io/docs/getting-started-guides/kubeadm/): 在不同操作系统，不同主机(Bare-Metal, AWS, GCE)条件下，快速部署集群。
- [*AWS EC2*](https://kubernetes.io/docs/getting-started-guides/aws/): 在aws上快速部署集群。
- [*Bare-Metal*](https://kubernetes.io/docs/getting-started-guides/centos/centos_manual_config/): 在物理机上手动部署。

可以参考[这个表格](https://kubernetes.io/docs/getting-started-guides/#table-of-solutions)选择适合您的场景的合适方案。

## 选择存储方案

容器不会保留在运行时生成的数据，job或者应用程序在容器中运行时生成的数据会在容器销毁时消失。为了完成分布式机器学习训练任务，需要有一个外部的存储服务来保存训练所需数据和训练输出。
常见的可选存储服务包括：

- [*NFS*](https://github.com/kubernetes/kubernetes/tree/master/examples/volumes/nfs): 可以将磁盘上某个目录共享给网络中其他机器访问。部署和配置比较简单，可以用于小量数据的验证。不提供分布式存储，高可用，冗余等功能。NFS的部署方法可以参考[这里](http://www.tecmint.com/how-to-setup-nfs-server-in-linux/)。
- [*GlusterFS*](http://gluster.readthedocs.io/en/latest/Quick-Start-Guide/Quickstart/): 网络分布式文件系统，可以在Kubernetes中按照[这个](https://github.com/kubernetes/kubernetes/tree/master/examples/volumes/glusterfs)例子使用。
- [*Ceph*](http://docs.ceph.com/docs/master/): 分布式文件系统，支持rbd，POSIX API接口(ceph fs)和对象存储API，参考[这里](https://kubernetes.io/docs/user-guide/volumes/#rbd)。
- [*MooseFS*](https://moosefs.com/documentation.html): 一个分布式的存储系统。需要先挂载到服务器Node上再通过kubernetes hostPath Volume挂载到容器中。

## 配置kubectl

### 安装kubectl
```
# OS X
curl -LO https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/darwin/amd64/kubectl

# Linux
curl -LO https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl

# Windows
curl -LO https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/windows/amd64/kubectl.exe
```

### 配置kubectl访问你的kubernetes集群

编辑`~/.kube/config`这个配置文件，修改`Master-IP`的地址。如果使用SSL认证，则需要配置`certificate-authority`和`users`中的用户证书。如果是使用非SSL方式访问（比如通过8080端口），也可以去掉这些证书的配置。
```
apiVersion: v1
clusters:
- cluster:
    certificate-authority: /path/to/ca.crt
    server: https://[Master-IP]:443
  name: minikube
contexts:
- context:
    cluster: minikube
    user: minikube
  name: minikube
current-context: minikube
kind: Config
preferences: {}
users:
- name: minikube
  user:
    client-certificate: /path/to/apiserver.crt
    client-key: /Users/wuyi/.minikube/apiserver.key
```
