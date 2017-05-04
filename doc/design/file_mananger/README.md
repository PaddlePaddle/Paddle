# Desgin doc: FileManager
## Objetive
在本文档中，我们设计说明了用户上传、下载、管理自己在PaddlePaddle Cloud上的文件所涉及到的模块和流程

<image src=./src/filemanager.png width=600>

## Module
### Client
Client提供用户管理本地或者远程文件的命令行程序。

- 路径参数:  
当用户输入一个命令的时候，一般需要指定路径参数。这里有两种路径参数：LocalPath 或者 PFSPath。

	LocalPath：代表本地的一个路径  
	PFSPath：代表PaddlePaddle Cloud上的一个路径。它需要满足类似这样的格式：`pfs://dir1/dir2`。路径必须要以`pds://`开始。

- 路径参数的顺序  
如果命令都有一个或者多个路径参数，那么一般第一个路径参数代表source，第二个路径参数代表destination。

- 支持的操作命令
	- [rm](cmd_rm.md)
	- [mv](cmd_mv.md)
	- [cp](cmd_cp.md)
	- [ls](cmd_ls.md)
	- [mkdir](cmd_mkdir.md)
	- [sync](cmd_sync.md)


### Ingress
- 在kubernets中运行
- 做Http转发、负载均衡
	- 注意配置session保持，以便来自一个用户的访问可以定向到一个固定的机器上，减少冲突写的机会。


### FileServer
功能说明:  
- gorpc写的HttpServer  
- 响应外部的REST API的请求  
- 在kubernets中运行  

REST API说明:

- file

```
GET /file: Get attribue of files
POST /file: Touch a file 
DELETE /file: Delete a File
```

- chunk
 
```
GET /file/chunk: Get a chunk info 
POST /file/chunk: Update a chunk
```
为什么有chunk的抽象：  
用户文件可能是比较大的，上传到Cloud或者下载到本地的时间可能比较长，而且在传输的过程中也可能出现网络不稳定的情况。为了应对以上的问题，我们提出了chunk的概念，一个chunk由所在的文件偏移、数据、数据长度及校验值组成。文件数据内容的上传和下载都是都过chunk的操作来实现的。由于chunk比较小（默认256K），完成一个传输动作的transaction的时间也比较短，不容易出错。

```
type Chunk struct {
	filePos int64
	checkSum uint32
	len     uint32
	data    []byte
}
```  

- dir

```
GET /dir: List all files in a directory
POST /dir: Touch a directory
DELETE /dir: Delete a directory
```


## 流程
### 关于文件权限
- 每一个用户在Cloud注册后可以申请分配用户空间，系统默认会在CephFS上分配一个固定大小（比如初始10G）的、有所有权限的volume，对用户而言就是自己的`home`目录。用户彼此之间的数据是隔离的、无法访问的。用户的空间大小第一期也不允许扩大。
- 公共数据集合放到一个单独的volume下，对所有外部用户只读。由于其被读取的可能比较频繁，需要提高其备份数，防止成为热点文件。

### 关于认证
> 通信各方都需要有各自的身份证。一个公司可以自签名一个CA身份证，并 且用它来给每个雇员以及每个程序签署身份证。这样，只要每台电脑上都预先安 装好公司自己的CA身份证，就可以用这个身份证验证每个雇员和程序的身份了。 这是目前很多公司的常用做法  

身份的认证来自于用户或者程序是否有crt标识身份，以及是否有可信的CA确认这个身份证是否有效。我们这里描述的crt涉及到两个部分，一个是Client端程序访问FileServer的crt，不妨称之为Client crt；另外一个是FileServer访问CephFS的crt，不妨称之为CephFS crt。

- Client和FileServer相互认证的办法   
`cloud.paddlepaddle.org`需要有自己的CA，FileServer和注册用户也要为其生成各自的私钥(key)、crt。这样用户把CA、自己的key和crt下载到本地后，Client程序可以用之和FileServer可以做相互的认证

- CephFS验证FileServer的身份的两种方法
	- 第一种：每一个用户都有自己单独的访问CephFS crt。
	用户访问其空间时，由FileServer读取它然后才可以在CephFS上完成操作。
	- 第二种：CephFS crt只有一个，也就是admin crt，拥有所有volume的读写权限。  
	FileServer从Client crt提取Client的身份（username），限制其可以操作的volume。 我们选择这种。

### 关于文件传输
文件传输的的关键在于需要Client端对比src和dst的文件chunks的checkSum是否保持一致，不一致的由Client Get或者Post chunk完成。藉由上述的方法完成断点的数据传输。 upload文件时，由于一个文件可以是多个FileServer可写的，存在冲突的机会，需要Client端在Post最后一个chunk的时候检查dest文件的MD5值是否和本地文件一致。

- 优化的方法:  

	- dst文件不存在时，可以没有Get的过程，只有Post。
	- 文件的chunks信息可以做cache，不用每次启动传输都去读和计算。这个由于比较复杂，第一期暂时不做。

- 小的技巧：

	- 可以用[Fallocate](https://golang.org/pkg/syscall/#Fallocate)生成sparse文件，让dst和src文件保持相同的大小。不同位置的chunk可以同时写入。

### 关于框架
准备拿出一点时间测试一下用[swagger-api](https://github.com/swagger-api/swagger-codegen)生成Client和FileServer的框架部分。如果框架生成好用，我们的精力就可以更多的放到逻辑本身上。

## 参考文档
- [Do you see tls?](https://github.com/k8sp/tls/blob/master/README.md)
- [s3](http://docs.aws.amazon.com/cli/latest/reference/s3/)
