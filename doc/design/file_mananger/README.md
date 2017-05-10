# FileManager设计文档
## 目标
在本文档中，我们设计说明了名为FileManager系统，方便用户管理存放到PaddlePaddle Cloud上的文件。   
主要功能包括：

- 提供常用的命令行文件管理命令管理文件
	- 支持的命令在[Here]	(./pfs/pfs.md)
- 支持大文件的断点上传、下载  

## 名词解释
- PFS：是Paddlepaddle cloud File System的简称，是对用户文件存储空间的抽象，与之相对的是Local File System。目前我们用CephFS来搭建。
- [CephFS](http://docs.ceph.com/docs/master/cephfs/)：一个POSIX兼容的文件系统。
- Chunk：逻辑划上文件分块的单位。
- [Ingress](https://kubernetes.io/docs/concepts/services-networking/ingress/)：提供七层协议的反向代理、基于粘性会话的负载均衡。
- CA：certificate authority<sup>[tls](#tls)</sup>
- CRT：CA signed certificate<sup>[tls](#tls)</sup>
- Key：用户私钥<sup>[tls](#tls)</sup>

## 模块

### 架构图
<image src=./src/filemanager.png width=900>

### PFSClient
- 功能： 详细的内容看[Here](./pfs/pfs.md)
	- 提供用户管理文件的命令
	- 用Golang写，可以跨平台执行

- 双向验证   
	PFSClient需要和Ingress之间做双向验证<sup>[tls](#tls)</sup>，所有用户需要首先在`cloud.paddlepaddle.org`上注册一下，申请用户空间，并且把系统生成的Key、CRT、CA下载到本地，然后才能使用PFSClient。
	
### Ingress
- 功能：  
	提供七层协议的反向代理、基于粘性会话的负载均衡功能。
	
- 透传用户身份的办法  
	Ingress需要把PFSClient的身份头传给FileServer，配置的方法参考[Here](http://www.integralist.co.uk/posts/clientcertauth.html#3)


### FileServer
FileServer是一个用GoRPC写的HTTPServer，提供[RESTful API](./RESTAPI.md)接口，接收处理PFSClient端的文件管理请求，并且把结果返回PFSClient端。

## 文件传输优化

### 分块文件传输
用户文件可能是比较大的，上传到Cloud或者下载到本地的时间可能比较长，而且在传输的过程中也可能出现网络不稳定的情况。为了应对以上的问题，我们提出了Chunk的概念，一个Chunk由所在的文件偏移、数据、数据长度及校验值组成。文件数据内容的上传和下载都是都过Chunk的操作来实现的。由于Chunk比较小（默认256K），完成一个传输动作完成的时间也比较短，不容易出错。PFSClient在传输完毕最后一个Chunk的时候检查desttination文件的MD5值是否和source文件一致。

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
当destination文件不存在或者大小和source文件不一致时，可以用[Fallocate](https://golang.org/pkg/syscall/#Fallocate)生成sparse文件，然后就可以并发写入多个Chunk。

### 覆盖不一致的部分
文件传输的的关键在于需要PFSClient端对比source和destination的文件Chunks的checksum是否保持一致，不一致的由PFSClient下载或者传输Chunk完成。这样已经传输成功的部分就不用重新传输了。

## 框架生成
用[swagger-api](https://github.com/swagger-api/swagger-codegen)生成Client和FileServer的框架部分，以便我们可以把更多的精力放到逻辑本身上。

## 参考文档
- <a name=tls></a>[TLS complete guide](https://github.com/k8sp/tls/blob/master/tls.md)
- [aws.s3](http://docs.aws.amazon.com/cli/latest/reference/s3/)
- [linux man document](https://linux.die.net/man/)
