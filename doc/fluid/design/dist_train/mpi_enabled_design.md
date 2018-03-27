#MPI-enabled PaddlePaddle Design doc
## Overview
We will introduce Open MPI API to PaddlePaddle, which can bring two benefits to PaddlePaddle:
1. Enable RDMA with PaddlePaddle, which bring high performance low latency networks.
2. Enable GPUDriect with PaddlePaddle, which bring highest throughput and lowest latency GPU read and write.

## Global Config
Launch the script using the 'mpirun' launcher, For example: ```mpirun -np 3 -hosts node1,node2,node3 python train.py```. By doing this, We can number the actors (trainer/pserver/master) whith o .. (n-1). The actor's number is the Rank of the calling process in group of comm (integer),  The MPI processes identify each other using an Rank ID. We have to create a mapping between PaddlePaddle's actors and there Rank ID, so that we can communicate with the correct destinations when using MPI operations.
    **We have to store the Rank ID and the mapping in global variables.**

#Utils
We will build mpi_send_recv_utils Class to unify package  interface about MPI Send and Receive.
```c++
#mpi send and receive utils
class Mpi_ISend {
    
}
class Mpi_IRecv {
    
}

class MPIUtils {
    public:
        const int GetRankID(const std::string& task_id);
        void InitMPI();
    private:
        std::map<std::string, int> name_to_id_;
}

```
```c++
class MPIServer {
    public:
        SetCond();
        ShutDown();
        WaitClientGet();
        reset();
        Push();
        SetScope();
        SetDevCtx();
        get();
}
```

## New OP
We won't replace all the gRPC requests to MPI requests,  the standard gRPC library is used for all administrative operations  and the MPI API will used to transfer tensor or selectRows to Pservers. Base of this idea, we create two new operators to handle requests and receives,  the two  operators are send_mpi_op and listenandserve_mpi_op. They are a little similar with [send_op](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/send_op.cc) and [listen_and_serv_op](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/listen_and_serv_op.cc).

### send_mpi_op
vary similar with send_op, we will replace grpc with mpi send service.
### listenandserve_mpi_op
vary similar with listen_and_serv_op, we will replace grpc with mpi receive service.
## Build args
Beause MPI or CUDA need hardware supported, so we will add some build args to control compiling.
**The specific arguments is under design**
## Execute args
Launch the script using the 'mpirun' launcher, For example: ```mpirun -np 3 -hosts node1,node2,node3 python train.py```.