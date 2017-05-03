# Desgin doc: FileManager
## Objetive
在本文档中，我们设计说明了用户上传、下载、管理自己在PaddlePaddle Cloud上的文件所涉及到的模块和流程
<image src=./src/filemanager.png width=600>

## Module
### Client
Client是用户操作的界面，支持的命令如下

- ls
 
```bash
ls
```

- cp

```bash
cp 
```

- sync

```bash
sync
```

- mv

```bash
mv
```


### Ingress
- 在kubernets中运行
- 做Http转发
- 注意配置session保持


### FileServer
FileServer是gorpc写的HttpServer, 用来接收Client的REST API的请求，自身由kubernets来管理。
REST API说明

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

- dir

```
GET /dir: List all files in a directory
POST /dir: Touch a directory
DELETE /dir: Delete a directory
```


## 流程
### cp
### 断点续传
### sync
