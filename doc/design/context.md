## Context Design

`Net` is the container and controller of a set of `Operator`. Each `Operator` in `Net` has a method called `Run`, which will take `Variable` (containing `Tensor`) from `Scope` to make computation. There will be single or several threads to execute operator's `Run` method.

Since the Tensor computation are executed by Eigen library, which needs an Eigen::GpuDevice type object as  parameter in a GPU card, and the GpuDevice parameter is constructed with an Eigen::CudaStreamDevice object. We need to set both a specific GpuID and CudaStream to create a Eigen::CudaStream object.

At the same time, some computation work will executed by cuBLAS or cuDNN library. Take cuBLAS library as an example, we have to acquire a cublasHandle which binds on a CudaStream to make computation. It's the same way as Eigen library does.

A `Context` is corresponded to a thread and holds runtime resources, such as CudaStream, cublasHandle and so on. And the `Run` method of `Operator` will take `Context` from a specific thread as a parameter to get necessary runtime resources.

The `Run` method of `Operator` is defined as follows:

```c++
Error Run(Scope* scope, Context* context);
```



The future DAGNet is run by multi-threads, and each thread will have its own Eigen::GpuDevice object binding on different CudaStream, so that multi-threads can run parallelly on a same GPU card.

And a specific thread will be in charge of copy(communication) work. The copy thread will only need to get a CudaStream from corresponding Context to make data copy between CPU/GPU or GPUs.

We can make a summary:

- Differnet GPU cards have different GpuID, and we can make data parallelism on multi-GPUs.
- Multi-threads can run a Net parallelly on a single GPU card, and each thread has one Context.
- There also exists one single thread executing a Net sequentially, and all computation and communication work will use the same Context.


Context is defined as follows(just using for unify class CUDAContext and class CPUContext):

```
class Context {};
```

### CUDAContext

CUDAContext is defined as follows:：


```c++                                   
class DeviceGuard {
 public:
  explicit DeviceGuard(int newDevice)
      : previous_(GetCurrentGPUID()) {
    if (previous_ != newDevice) {
      cudaError_t err = cudaSetDevice(newDevice);
      PADDLE_ASSERT(err == cudaSuccess);
    }
  }

  ~DeviceGuard() {
    cudaError_t err = cudaSetDevice(previous_);
    PADDLE_ASSERT(err == cudaSuccess);
  }

 private:
  int previous_;
};

class CUDAContext : public Context{
public:
  explicit CDUAContext(const int gpu_id) : gpu_id_(gpu_id) {
    DeviceGuard(gpu_id_);
    cudaError_t err = cudaStreamCreate(&stream_);
    PADDLE_ASSERT(err == cudaSuccess);
    
    eigen_stream_ = new Eigen::CudaStreamDevice(&stream_);
    eigen_handle_ = new Eigen::GpuDevice(eigen_stream_);    
  }
  
  void Wait() {
    cudaError_t err = cudaStreamSynchronize(stream_);
    PADDLE_ASSERT(err == cudaSuccess);
  }
  
  cudaStream_t GetStream() {
    return stream_;
  }
  
  Eigen::GpuDevice GetEigenHandle() {
    return *eigen_handle_;
  }
  
  cublasHandle_t GetCuBLASHandle() {
    if (!blas_handle_) {
      DeviceGuard guard(gpu_id_);      
      cudaError_t err = cublasCreate(&blas_handle_);
      PADDLE_ASSERT(err == CUBLAS_STATUS_SUCCESS);
      cudaError_t err = cublasSetStream(blas_handle_, stream_);
      PADDLE_ASSERT(err == cudaSuccess);       
    }        
    return blas_handle_;
  }
  
  cudnnHandle_t GetCuDNNHandle() {
    if (!dnn_handle_) {
      DeviceGuard guard(gpu_id_);
      cudaError_t err = cudnnCreate(&dnn_handle_);
      PADDLE_ASSERT(err == CUDNN_STATUS_SUCCESS);
      cudaError_t err = cudnnSetStream(dnn_handle_, stream_);
      PADDLE_ASSERT(err == cudaSuccess); 
    }
    return dnn_handle_;
  }
  
  curandGenerator_t GetCuRandHandle() {
    if (! rand_handle_) {
      DeviceGuard guard(gpu_id_);
      cudaError_t err = curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT);
      PADDLE_ASSERT(err == CURAND_STATUS_SUCCESS);
      cudaError_t err = curandSetPseudoRandomGeneratorSeed(curand_generator_, random_seed_);
      PADDLE_ASSERT(err == CURAND_STATUS_SUCCESS);
      cudaError_t err = curandSetStream(curand_generator_, stream_);
      PADDLE_ASSERT(err == cudaSuccess);      
    }
    return rand_handle_;
  }
  
  ~CUDAContext() {
    Wait();
    
    if (blas_handle_) {
      cudaError_t err = cublasDestroy(blas_handle_);
      PADDLE_ASSERT(err == CUBLAS_STATUS_SUCCESS);
    }
    
    if (dnn_handle_) {
      cudaError_t err = cudnnDestroy(dnn_handle_);
      PADDLE_ASSERT(err == CUDNN_STATUS_SUCCESS);
    }
    
    if (rand_handle_) {
      cudaError_t err = curandDestroyGenerator(rand_handle_);
      PADDLE_ASSERT(err == CURAND_STATUS_SUCCESS);
    }
    
    delete eigen_stream_;
    delete eigen_handle_;
    cudaError_t err = cudaStreamDestroy(stream_);
    PADDLE_ASSERT(err == cudaSuccess);
  }

private:
  int gpu_id_;
  cudaStream_t stream_;
  
  Eigen::CudaStreamDevice* eigen_stream_;
  Eigen::GpuDevice* eigen_handle_;
  
  cublasHandle_t blas_handle_{nullptr};
  
  cudnnHandle_t dnn_handle_{nullptr};
  
  int random_seed_;
  curandGenerator_t rand_handle_{nullptr};
}；
```

### CPUContext

CPUContext is defined as follows:

```c++
class CPUContext : public Context{
  Eigen::DefaultDevice GetEigenHandle() {
    if (!eigen_handle_) {
      eigen_handle_ = new Eigen::DefaultDevice();
    }
    return *eigen_handle_;
  }
private:
  Eigen::DefaultDevice* eigen_handle_{nullptr};  
};
```
