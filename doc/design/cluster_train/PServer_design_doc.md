# Design Doc: Parameter Server Process

Parameter Server process 是Paddle中负责模型的存储，更新和模型分片一致性的组件，在整个系统中的作用请参考 [distributed training design doc](./README.md) ，本文档包含PServer，PClient，PServerContoller等，涉及到的配置参数均使用大写字母

<img src="src/paddle-model-sharding.png" width="500"/>

## 术语

- PServer: Parameter Server 服务器
- PClient: Parameter Server Client
- PServerController：PServer管理员，启动Server，动态扩容，容灾等
- model: 指深度学习训练之后得到的所有参数，使用这个神经网络可以完成对新数据的预测
- parameters: 神经网络中的参数，包括权重w和偏置b。一个神经网络的模型由大量的参数组成
- shard: 分片，通常指将一个整体拆分成多份的其中的一份。
- parameter block: 多个parameter block构成一个model shard(现存的model并行策略是parameter block based，在新架构中继续沿用)
- 单点故障: 任意时刻只可能同时有一台服务器故障。由于集群中同时存在两台机器故障的概率极低（（平均故障率*平均故障修复时间）^2）只对特殊在线系统考虑两台以上同时故障的容灾。

##  PServer

PServer负责:

模型存储，模型更新，注册服务并监听端口事件，PServer个数的动态扩张收缩，负责序列化传输数据。

发送接收数据和命令都使用rpc 接口，例如golang rpc

```c++
class Evaluator;
class ParameterUpdater;
class DeviceSet;
template<PKEY, PVALUE>
class PServer {
  class ParameterSegments {
  PKEY key; // param_id;
  
  ...
	}
RWLock lock;
int32_t serverId;
PServerConfig config; // start Pserver config 
  
// part 1, store model, store model in device, e.g gpu, cpu memory
// compute resource 
// treat thread, memory as devices
syncThreadPool threadPool;
Device **store_pool[SHARD_NUM];  // memory and gpu memory，2d pointer store a vector of shard pointer. each shard should be unordered_map<ParameterSegments> parameterMap; 

GradientMachine *gmbase; // gradient machine implement forward backward interface, hidden the detail of communication of devices, such as GPUMerge, multithread Merge , see multineuralnet, recurrnet neuralnet, etc
  
//register operations service,  used between matrix, vectors ooperation
//operation function name : operationFunction 
unorderedmap<string, operationFunc> serviceRegisted;
// e.g example from the code in paddle
1, regist service function in PServer
serviceRegisted.insert("PSERVER_OP_utu", OpFunc_UTU);
2, pack rpc call with method_name="PSERVER_OP_utu", PServer will check the service map and execute OpFunc in Parallel
   OpFuncName = request.request_method_name;
   auto OpFuncRpcCalled = serviceRegisted.find(OpFuncName)
   CHECK(OpFuncRpcCalled);
   parallelExec()
     or
   doOperation(OpFuncRpcCalled)
3, pack response and send to Client
   response = getResponse()
   response.set_result(res)
   serilize/archive to binary blob, send response by rpc call 
public:
  int32_t init();
  int32_t isStartedAndAlive(); // for PServerController, check status
  
  // part 2: update parameter
  // *ONLY* use this interface execute the callback
private:
// apply update
ParameterUpdater *updatebase;

  typedef std::function<void()> Callback;  // function callback
  void exec();
  void parallelExec();
  or
  
  void doOperation(PrepareOperation& ops, ...); // operator topology
  void doMultipleOperation(PrepareOperation& ops, ...);
  //TODO: op execute need more detail here
  
  // part 4: checkpoint, ignore the difference of save time between PServer nodes.
  // see hash ring, when there is failed worker, kubernates start a new worker and insert into hashring.
  hashring registerWorker;
  
  int32_t saveCheckPoint() {
    1, counter match save checkpoint condition, grab the RWLock;
    2, start new thread, generate unique UUID, write to pfs(filesystem), (TODO: Stop update and wait?)
    3, write etcd `/checkpoint/pserver_id : {"uuid": [UUID], "md5", "MD5 sum", "timestamp": xxxx}`
    4, delete earlier checkpoint not equal to UUID
    5, release lock, wait write thread join; 
    return SUCCESS;
  }
  int32_t recoveryFromCheckPoint() {
    getUUIDFrometcd(); 
    tryLoadCheckPoint();
    PServerController call start interface.
    return SUCCESS;
  }
  
private:
// metrics, evaluate the model in runtime
// every node send runtime statistics to evaluatorServer during training/testing. when training Pass finish(or event handler notify), trainer leader(e.g node_id=0) send rpc call to evaluatorServer process then produce result. 
//Evaluator base class, for example, AUC, LOSS, AVERAGE 
//evalbase->sendAsync(EVAL_DATA_STRUCT)
//evaluatorServer as standalone thread, can be used in jupyter notebook
Evaluator *evalbase;  
  
//part 6 : auto scaling 
a. Trainer/worker auto scaling insert or remove 
rehash key based on Pserver, see PClient Part
 
} // PServer


class ParameterUpdater {
Optimizer *base;
}

class SparseParameterUpdater{
  //TODO: need to discussed
}
class SparseParameterUpdater {
  
}
class SGDOptimizer : Optimizer {
  ...
}
class ASGDOptimizer : Optimizer {
  ...
}
class OWLQNOptimizer : Optimizer {
  ...
}
```

<img src="src/hashring.png" width="300"/>

Optimizer需要支持的优化算法

L-BFGS，owlqn，ftrl, TODO：在Paddle中owlqn等需要参数更新方式不同，支持接口是否相同？

sgd (momentum, adagram, adadelta, adam)，pass based

async-sgd



## PClient

PClient功能是否已经包含在trainer中？PClient 负责parameter balancer，打包rpc请求转发PServer。

```c++
class ParameterPartitioner;
template<PKEY, PVALUE>
PClient {
public:
  // pack request as rpc call and serilize/archive sending data 
  void eventHandler();
private:
  // use param_id and node_id as hash key, balance parameter between shard and PServers
  ParameterPartitioner partitioner;
}
template<PKEY>
class ParameterPartitioner {
  hash(param_id, node_id) // impl hash function generate evenly distributed shard_id/PServerid, when auto scaling of PServer, then store 
    
// auto scaling, do not implement in v1, 
//TODO: need more detail
  rehash(param_id, node_id); // generate new server hash id for each parameter
  
}
```



## RPCServer

接口类，屏蔽rpc实现，方便移植rpc lib

```c++
// Used for request, package up request into binary
struct RpcRequest {
    uint64_t _request_id;
    int32_t _src_id;  //node_id
    int32_t _target_id; //node_id
    std::string _target_method_name; 
    std::string _src_method_name;
    BinaryArchive _args;
    static uint64_t s_buffer_size;
}
// Used for response, package up response into binary
struct RpcResponse {
public:
    uint64_t _request_id;
    int32_t _target_id;
    std::string _target_method_name;
    BinaryArchive _archive;
    int32_t _error_code;
```



```c++
class AsyncRPCServer {
  static createRequest(RPCRequest*, RPCResponse);
  static AsyncRPCServer& singleton();
  // send rpc call asynchronize
  void send_async(); 
  void send_sync();
  RPCImpl *rpcimpl;
}
```

## PServerController

根据ParameterServer参数创建和管理PServer instance，从命令行读取参数，从etcd读取参数，运行开始将存活的PServer instance配置存储在etcd中

- 启动和运行参数包括：

`/PS_DESIRED`:, 启动PServer instance个数，etcd存储格式 `/PS_DESIRED:3`

`/ROOT_PORT`：显式指定根端口，PServer端口从PORT+1开始，直到找到可用端口，例如ROOT_PORT=8000, 则PServer0_Port=8000，PServer0_Port=8001,…，当前存在的PServer实例配置以etcd实时存储为准

etcd存储格式 `ROOT_PORT:8000, /PS/0:8000，/PS/1:8001 `

`/CHECKPOINT_PERIOD`:PServer运行保存快照存储的时间间隔，default filled

`/CHECKPOINT_DIR`:保存快照的路径，default filled

- 创建PServer接口

```c++
int32_t loadConfig(fromCLi);
int32_t loadConfig(fromEtcdDir);
// create PServer fron scratch or recovery from config  
static PServer* create(PServerConfig& );
// create PServer in fault tolenrant, recovery from checkpoint 
static PServer* create(const char* checkpoint_dir);
```

- 管理PServer实例

```c++
int32_t start(); // start PServer
int32_t wait();  //wait join
int32_t countAlive(); // count alive instance
```