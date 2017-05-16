# FileManager设计文档
## 目标
在本文档中，我们设计说明了名为FileManager系统，方便用户上传自己的训练数据以进行分布式训练

主要功能包括：

- 提供常用的命令行管理命令管理文件和目录
- 支持大文件的断点上传、下载  

## 名词解释
- PFS：是`Paddlepaddle cloud File System`的缩写，是对用户文件存储空间的抽象，与之相对的是local filesystem。目前我们用CephFS来搭建。
- [CephFS](http://docs.ceph.com/docs/master/cephfs/)：一个POSIX兼容的文件系统。
- Chunk：逻辑划上文件分块的单位。

## 模块
### 架构图
<image src=./src/filemanager.png width=900>

### PFSClient
- 功能： 详细设计[link](./pfs/pfsclient.md)
	- 提供用户管理文件的命令
	- 需要可以跨平台执行

- 双向验证   
	PFSClient需要和Ingress之间做双向验证<sup>[tls](#tls)</sup>，所以用户需要首先在`cloud.paddlepaddle.org`上注册一下，申请用户空间，并且把系统生成的CA(certificate authority)、Key、CRT(CA signed certificate)下载到本地，然后才能使用PFSClient。
		
### [Ingress](https://kubernetes.io/docs/concepts/services-networking/ingress/)
- 功能：  
	提供七层协议的反向代理、基于粘性会话的负载均衡功能。
	
- 透传用户身份的办法  
	Ingress需要把PFSClient的身份信息传给PFSServer，配置的方法参考[link](http://www.integralist.co.uk/posts/clientcertauth.html#3)

### PFSServer
PFSServer提供RESTful API接口，接收处理PFSClient端的文件管理请求，并且把结果返回PFSClient端。

RESTful API

- /api/v1/files
	- `GET /api/v1/files`: Get metadata of files or directories.
	- `POST /api/v1/files`: Create files or directories.
	- `PATCH /api/v1/files`: Update files or directories.
	- `DELETE /api/v1/files`: Delete files or directories.

- /api/v1/file/chunks
	- `GET /api/v1/storage/file/chunks`: Get chunks's metadata of a file.

- /api/v1/storage/files
	- `GET /api/v1/storage/files`: Download files or directories.
	- `POST /api/v1/storage/files`: Upload files or directories.

- /api/v1/storage/file/chunks
	- `GET /api/v1/storage/file/chunks`: Download chunks's data.
	- `POST /api/v1/storage/file/chunks`: Upload chunks's data.

## 文件传输优化

### 分块文件传输
用户文件可能是比较大的，上传到Cloud或者下载到本地的时间可能比较长，而且在传输的过程中也可能出现网络不稳定的情况。为了应对以上的问题，我们提出了Chunk的概念，一个Chunk由所在的文件偏移、数据、数据长度及校验值组成。文件的上传和下载都是通过对Chunk的操作来实现的。由于Chunk比较小（默认256K），完成一个传输动作完成的时间也比较短，不容易出错。PFSClient需要在传输完毕最后一个Chunk的时候检查destination文件的MD5值是否和source文件一致。

一个典型的Chunk如下所示：

```
type Chunk struct {
	fileOffset int64
	checksum uint32
	len     uint32
	data    []byte
}
```  

### 生成sparse文件
当destination文件不存在或者大小和source文件不一致时，可以用[Fallocate](https://Go.org/pkg/syscall/#Fallocate)生成sparse文件，然后就可以并发写入多个Chunk。

### 覆盖不一致的部分
文件传输的的关键在于需要PFSClient端对比source和destination的文件Chunks的checksum是否保持一致，不一致的由PFSClient下载或者传输Chunk完成。这样已经传输成功的部分就不用重新传输了。

## 用户使用流程
参考[link](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/cluster_train/data_dispatch.md)

## 框架生成
用[swagger](https://github.com/swagger-api/swagger-codegen)生成PFSClient和PFSServer的框架部分，以便我们可以把更多的精力放到逻辑本身上。

## 参考文档
- <a name=tls></a>[TLS complete guide](https://github.com/k8sp/tls/blob/master/tls.md)
- [aws.s3](http://docs.aws.amazon.com/cli/latest/reference/s3/)
- [linux man document](https://linux.die.net/man/)
