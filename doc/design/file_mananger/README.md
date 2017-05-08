# FileManager设计文档
## 名词解释
- PFS：是Paddle cloud File System的简称。与之相对的是Local File System。
- FileServer：接收用户管理文件命令的服务端
- FileManger：用户管理自己自己在PFS文件上的系统称为FileManager
- CephFS：是个POSIX 兼容的文件系统，它使用 Ceph 存储集群来存储数据.
- Chunk：逻辑划分文件的一个单位，用于文件分块上传或者下载。
- Ingress：Layer 7 Load Balancer

## 目标
在本文档中，我们设计说明了用户上传、下载、管理自己在PaddlePaddle Cloud上的文件所涉及到的模块和流程。架构图如下所示：

<image src=./src/filemanager.png width=8900>



## PFS Client
- 提供用户管理Cloud文件的命令
- 用Golang写，可以跨平台执行

命令的详细内容看[Here](./pfs/pfs.md)


### Ingress
- 在kubernets中运行
- 做HTTP转发、负载均衡
	- 注意配置session保持，以便来自一个用户的访问可以定向到一个固定的机器上，减少冲突写的机会。


## FileServer
功能说明:  
- goRPC写的HTTPServer  
- 响应外部的REST API的请求  
- 在kubernetes中运行  
- [RESTAPI](./RESTAPI.md)接口

## 文件传输
### 文件权限
- 每一个用户在Cloud注册后可以申请分配用户空间，系统默认会在CephFS上分配一个固定大小（比如初始10G）的、有所有权限的volume，对用户而言就是自己的`home`目录。用户彼此之间的数据是隔离的、无法访问的。用户的空间大小第一期也不允许扩大。
- 公共数据集合放到一个单独的volume下，对所有外部用户只读。由于其被读取的可能比较频繁，需要提高其备份数，防止成为热点文件。

### 用户认证
身份的认证来自于用户或者程序是否有crt标识身份，以及是否有可信的CA确认这个身份证是否有效。我们这里描述的crt涉及到两个部分，一个是Client端程序访问FileServer的crt，不妨称之为Client crt；另外一个是FileServer访问CephFS的crt，不妨称之为CephFS crt。

- Client和FileServer相互认证的办法   
`cloud.paddlepaddle.org`需要有自己的CA，FileServer和注册用户也要为其生成各自的私钥(key)、crt。这样用户把CA、自己的key和crt下载到本地后，Client程序可以用之和FileServer可以做相互的认证。

- CephFS验证FileServer的身份的方法  
	CephFS crt只有一个，也就是admin crt，拥有所有volume的读写权限。  FileServer从Client crt提取Client的身份（username），限制其可以操作的volume。 

### 分块文件上传
用户文件可能是比较大的，上传到Cloud或者下载到本地的时间可能比较长，而且在传输的过程中也可能出现网络不稳定的情况。为了应对以上的问题，我们提出了chunk的概念，一个chunk由所在的文件偏移、数据、数据长度及校验值组成。文件数据内容的上传和下载都是都过chunk的操作来实现的。由于chunk比较小（默认256K），完成一个传输动作完成的时间也比较短，不容易出错。

一个典型的chunk如下所示：

```
type Chunk struct {
	fileOffset int64
	checksum uint32
	len     uint32
	data    []byte
}
```  

### 文件传输的优化
文件传输的的关键在于需要Client端对比source和destination的文件chunks的checkSum是否保持一致，不一致的由Client Get或者Post chunk完成。藉由上述的方法完成断点的数据传输。 upload文件时，由于一个文件可以是多个FileServer可写的，存在冲突的机会，需要Client端在Post最后一个chunk的时候检查dest文件的MD5值是否和本地文件一致。

- 优化的方法:  

	- dst文件不存在时，可以没有Get的过程，只有Post。

- 小的技巧：

	- 可以用[Fallocate](https://golang.org/pkg/syscall/#Fallocate)生成sparse文件，让dst和src文件保持相同的大小。不同位置的chunk可以同时写入。


## 框架生成
用[swagger-api](https://github.com/swagger-api/swagger-codegen)生成Client和FileServer的框架部分，以便我们可以把更多的精力放到逻辑本身上。

## 参考文档
- [TLS complete guide](https://github.com/k8sp/tls/blob/master/README.md)
- [aws.s3](http://docs.aws.amazon.com/cli/latest/reference/s3/)
- [linux man document](https://linux.die.net/man/)
