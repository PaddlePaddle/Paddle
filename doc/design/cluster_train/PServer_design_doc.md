# Design Doc: Parameter Server

ParameterServer 是Paddle中分布式训练更新模型的组件，在整个系统中的作用请参考 [distributed training design doc](./README.md) ，本文档包含ParameterServer，ParameterClient，ParameterServerContoller等，涉及到的配置参数均使用大写字母

<img src="src/paddle-model-sharding.png" width="500"/>

## 术语

- ParameterServer: ParameterServer Server，负责模型存储，调用分布式更新，响应ParameterClient请求
- ParameterClient: ParameterServer Client，负责均衡ParameterServer请求，打包并转发RPC请求
- ParameterServerController：负责启动Server，动态扩容，容灾等
- Tensor: 一个NDArray结构，Trainer与ParameterServer, ParameterClient交互的基本数据结构
- shard: 全量模型在某个ParameterServer上的局部分片，通常指将一个模型整体拆分成多份的其中的一份。
- parameter block: 多个parameter block构成一个shard(现存的model并行策略是parameter block based，在新架构中继续沿用)

##  ParameterServer

ParameterServer负责以下功能:

1、模型存储，2、注册服务并监听端口事件，3、ParameterServer instance故障恢复，其中Trainer个数的动态扩张收缩，4、负责序列化传输数据。

发送接收调用都使用RPC 接口，见下文中的RPCServer，例如使用Go RPC实现对应的接口

```c++

/* Because there is no Tensor data structure right now,  \ 
optimizer in ParameterServer does not need the Tensor shape,  \
we just define Vector as Tensor, should be replace with `real Tensor` \
after the refactoring finish. */
typedef /*Vector*/ Tensor<DIM=1, PVALUE>;
template<PKEY, PVALUE>
class ParameterServer {

RWLock lock;
/* ParameterServer_id used by checkpoint */
int32_t ParameterServer_id;
/* start ParameterServer config, should be persist in ectd for ParameterServer node recovery */
ParameterServerConfig config;   

// part 1: store model in ParameterServer
// use Tensor as store fundamental unit

syncThreadPool threadPool;

/*
when init() calls, create SHARD_NUM Shard_Store;
parameters:
SHARD_NUM : int, store in ParameterServerConfig.
	model shard in one ParameterServer node;
*/
typedef unordered_map<block_id, Tensor<PVALUE>> Shard_Store;
/* 2d pointer store a vector of shard pointer. each shard should be unordered_map<block_id, Tensor> parameterMap; 
block_id is the parameter block id, after scling with the Sclicer(see ParameterClient), slice parameter Matrix generate parameter block; 
*/
Shard_Store **store_pool;  
 
public:
  /* init */
  int32_t init();
  /* used by ParameterServerController check status */
  bool is_started;
  int32_t isStartedAndAlive(); 
  
  /* deserilize/unarchive sending data */
  void PullParameters_process_handler(RpcRequest, RpcResponse);
  /* deserilize/unarchive updating data, need to call setUpdater first time */
  void UpdateParameters_process_handler(RpcRequest, RpcResponse);
  /*  get Parameters thread for parallel */
  int32_t thread_hPullParameters(int32_t thread_id, <map<string/*pname*/>*Tensor params);
  /* set Parameters thread for parallel */
  int32_t thread_UpdateParameters(int32_t thread_id, <map<string/*pname*/>*Tensor params);
  
  /* set updater/optimizer */
  int32_t set_updater(updater_name) {
   updatebase = updater;
  }
private:
// apply update
ParameterUpdater *updatebase;
 /* part 2 : checkpoint, ignore the difference of save time between ParameterServer nodes. */
  int32_t saveCheckPoint() {
    /*
    1, counter match save checkpoint condition, grab the RWLock;
    2, start new thread, generate unique UUID, write to pfs(filesystem), (TODO: Stop update and wait?)
    3, write etcd `/checkpoint/ParameterServer_id : {"uuid": [UUID], "md5", "MD5 sum", "timestamp": xxxx}`
    4, delete earlier checkpoint not equal to UUID
    5, release lock, wait write thread join;  */
    return SUCCESS;
  }
  int32_t recoveryFromCheckPoint() {
    /*
    1, getUUIDFrometcd(); 
    2, tryLoadCheckPoint();
    3, ParameterServerController call start interface. */
    return SUCCESS;
  }
  
private:

  
//part 3 : auto scaling of ParameterServers 
/* part 3.a. Trainer/worker auto scaling insert or remove during training */
 unordered_map<string/*trainer name*/, Trainer*>

} // ParameterServer


class ParameterUpdater {
Optimizer *base;
}

class SparseParameterUpdater{
  //TODO: need to discussed
}
class SparseParameterUpdater {
  
}
/* 目前支持SGD(Stochastic Gradient Descent) 类算法，不支持OWLQN (Orthant-Wise Limited-memory Quasi-Newton)等算法
   SGD (SGD, Momentum, Adam)
   async-SGD
*/
class SGDOptimizer : Optimizer {
  ...
}
class ASGDOptimizer : Optimizer {
  ...
}
```



## ParameterClient

ParameterClient 负责均衡ParameterServer请求，打包rpc请求转发ParameterServer。

```c++
/* named the block slicer, cut Parameter into blocks, and deletermine its ParameterServer_id(shard_id)*/
class Slicer;
template<PKEY, PVALUE>
ParameterClient {
public:
/* get Parameters */
int32_t PullParameters(<map<string/*pname*/>*Tensor params);
/* set Parameters */
int32_t UpdateParameters(<map<string/*pname*/>*Tensor params)
  
/* pack request as rpc call and serilize/archive receive data */
void PullParameters_rpc_handler(RpcRequest, RpcResponse);
/* pack request as rpc call and serilize/archive sending data */
void UpdateParameters_rpc_handler(RpcRequest, RpcResponse);

private:
/* use param_id and node_id as hash key, balance parameter between shard and ParameterServers */
  Slicer _slice;


}

template<PKEY>
class Slicer {
  /* impl hash function generate evenly distributed shard_id/ParameterServerid, when auto scaling of ParameterServer, then store  */
  hash(param_id, node_id) 
}
```



## RPCServer

接口类，屏蔽rpc实现，方便移植rpc lib

```c++
/* RpcRequest Header, Used for request, package up request into RpcImpl Call */
struct RpcRequest {
    uint64_t _request_id;                // request_id 
    int32_t _src_id;                     // request source node_id or process_id, is unique in k8s.
    int32_t _target_id;                  // request target. same as before
    std::string _target_method_name;     // Command bewteewn two nodes . e.g 
  /*format example, [COMMAND_NODE]
    SetUpdater_server_id;
    PullParameters_server_id;   
    UpdateParameters_server_id;
    SaveCheckPoint_server_id;
    
    RegisterTrainer_trainer_id;
    DeregisterTrainer_trainer_id;
	....
	DoOperation_server_id; [COMMAND_extension]
  */
    BinaryArchive _args;                 // other arguments for extension  
    static uint64_t s_buffer_size;      
}
// Used for response, package up response into binary
struct RpcResponse {
public:
    uint64_t _request_id;
    int32_t _target_id;
    std::string _target_method_name;
    BinaryArchive _archive;
    int32_t _error_code;                  // return status code
```



```c++
class RPCServer {
  /* ONLY use this method create RPC Calls */
  void static createRequest(RPCRequest*, RPCResponse);  
  static RPCServer& singleton();
  // send rpc call asynchronize
  void Send(bool sync, ...);
  RPCImpl *rpcimpl;
}
```

## ParameterServerController

根据ParameterServer参数创建和管理ParameterServer instance，从命令行读取参数，从etcd读取参数，运行开始将存活的ParameterServer instance配置存储在etcd中

- 启动和运行参数包括：

  这部分参数都会存储于etcd中，用于自动扩容和容灾，运行可以从命令行读取，也可以从etcd读取

`/PS_DESIRED`:, 启动ParameterServer instance个数，etcd存储格式 `/PS_DESIRED:3`

`/ROOT_PORT`：显式指定根端口，ParameterServer端口从PORT+1开始，直到找到可用端口，例如ROOT_PORT=8000, 则ParameterServer0_Port=8000，ParameterServer0_Port=8001,…，当前存在的ParameterServer实例配置以etcd实时存储为准

etcd存储格式 `ROOT_PORT:8000, /PS/0:8000，/PS/1:8001 `

`

```c++
/* ParameterServer runtime configuration, used by recovery/rescaling */
class ParameterServerConfig {
...
/* set by trainer process */
/* CHECKPOINT_PERIOD:ParameterServer运行保存快照存储的时间间隔，default filled */
/* CHECKPOINT_DIR:保存快照的路径，default filled */
string CHECKPOINT_DIR:
int64_t CHECKPOINT_PERIOD;
}
```
- 创建ParameterServer接口

```c++
int32_t loadConfig(fromCLi);
int32_t loadConfig(frometcdDir);
// create ParameterServer fron scratch or recovery from config  
static ParameterServer* create(ParameterServerConfig& );
// create ParameterServer in fault tolenrant, recovery from checkpoint 
static ParameterServer* create(const char* checkpoint_dir);
```

- 管理ParameterServer实例

```c++
int32_t start();   // start ParameterServer
int32_t wait();    //wait join
int32_t countAlive(); // count alive instance
```



