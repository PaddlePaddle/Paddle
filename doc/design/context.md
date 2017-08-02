## Context Design

`Net` is the container and controller of a set of `Operator`. Each `Operator` in `Net` has a method called `Run` to make computation. The `Run` method of `Operator` is defined as follows:

```c++
Error Operator::Run(OpContext* context);
```

And the `OpContext` is defined as follows:

```c++
Struct OpContext {
  Scope* scope;
  DeviceContext* device_context;
};
```

The `Run` method will take `Variable` (containing `Tensor`) from `Scope` to make computation on certain `DeviceContext`. `DeviceContext` provides necessary runtime resources for computation, including CudaStream, cublasHandle and so on.

Generally speaking, the Tensor computation are executed by Eigen library, which needs an Eigen::GpuDevice type object as parameter in a GPU card, and the GpuDevice parameter is constructed with an Eigen::CudaStreamDevice object. We need to set both a specific GpuID and CudaStream to create a Eigen::CudaStream object.

At the same time, some computation work will executed by cuBLAS or cuDNN library. Take cuBLAS library as an example, we have to acquire a cublasHandle which binds on a CudaStream to make computation. It's the same way as Eigen library does.

`DeviceContext` is defined as follows(just using for unify class `CudaDeviceContext` and class `CpuDeviceContext`):

```
class DeviceContext {
  virtual ~Context() {}
};
```

### CudaDeviceContext

CudaDeviceContext is defined as follows:


```c++                                   
class DeviceGuard {
 public:
  explicit DeviceGuard(GPUPlace new_place)
      : previous_(GetCurrentGPUID()) {
    if (previous_ != new_place) {
      cudaError_t err = cudaSetDevice(new_place.device);
      PADDLE_ENFORCE(err == cudaSuccess);
    }
  }

  ~DeviceGuard() {
    cudaError_t err = cudaSetDevice(previous_.device);
    PADDLE_ENFORCE(err == cudaSuccess);
  }

 private:
  GPUPlace previous_;
};

class CudaDeviceContext : public DeviceContext{
public:
  explicit CDUAContext(const GPUPlace gpu_place) : gpu_place_(gpu_place) {
    DeviceGuard(gpu_place_);
    cudaError_t err = cudaStreamCreate(&stream_);
    PADDLE_ENFORCE(err == cudaSuccess);
    
    eigen_stream_ = new Eigen::CudaStreamDevice(&stream_);
    eigen_handle_ = new Eigen::GpuDevice(eigen_stream_);    
  }
  
  void Wait() {
    cudaError_t err = cudaStreamSynchronize(stream_);
    PADDLE_ENFORCE(err == cudaSuccess);
  }
  
  cudaStream_t stream() {
    return stream_;
  }
  
  Eigen::GpuDevice eigen_handle() {
    return *eigen_handle_;
  }
  
  cublasHandle_t cublas_handle() {
    if (!blas_handle_) {
      DeviceGuard guard(gpu_place_);
      cudaError_t err = cublasCreate(&blas_handle_);
      PADDLE_ENFORCE(err == CUBLAS_STATUS_SUCCESS);
      cudaError_t err = cublasSetStream(blas_handle_, stream_);
      PADDLE_ENFORCE(err == cudaSuccess);
    }        
    return blas_handle_;
  }
  
  cudnnHandle_t cudnn_handle() {
    if (!dnn_handle_) {
      DeviceGuard guard(gpu_place_);
      cudaError_t err = cudnnCreate(&dnn_handle_);
      PADDLE_ENFORCE(err == CUDNN_STATUS_SUCCESS);
      cudaError_t err = cudnnSetStream(dnn_handle_, stream_);
      PADDLE_ENFORCE(err == cudaSuccess);
    }
    return dnn_handle_;
  }
  
  curandGenerator_t curand_handle() {
    if (! rand_handle_) {
      DeviceGuard guard(gpu_place_);
      cudaError_t err = curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT);
      PADDLE_ENFORCE(err == CURAND_STATUS_SUCCESS);
      cudaError_t err = curandSetPseudoRandomGeneratorSeed(curand_generator_, random_seed_);
      PADDLE_ENFORCE(err == CURAND_STATUS_SUCCESS);
      cudaError_t err = curandSetStream(curand_generator_, stream_);
      PADDLE_ENFORCE(err == cudaSuccess);
    }
    return rand_handle_;
  }
  
  ~CUDAContext() {
    Wait();
    
    if (blas_handle_) {
      cudaError_t err = cublasDestroy(blas_handle_);
      PADDLE_ENFORCE(err == CUBLAS_STATUS_SUCCESS);
    }
    
    if (dnn_handle_) {
      cudaError_t err = cudnnDestroy(dnn_handle_);
      PADDLE_ENFORCE(err == CUDNN_STATUS_SUCCESS);
    }
    
    if (rand_handle_) {
      cudaError_t err = curandDestroyGenerator(rand_handle_);
      PADDLE_ENFORCE(err == CURAND_STATUS_SUCCESS);
    }
    
    delete eigen_stream_;
    delete eigen_handle_;
    cudaError_t err = cudaStreamDestroy(stream_);
    PADDLE_ENFORCE(err == cudaSuccess);
  }

private:
  GPUPlace gpu_place_;
  cudaStream_t stream_;
  
  Eigen::CudaStreamDevice* eigen_stream_;
  Eigen::GpuDevice* eigen_handle_;
  
  cublasHandle_t blas_handle_{nullptr};
  
  cudnnHandle_t dnn_handle_{nullptr};
  
  int random_seed_;
  curandGenerator_t rand_handle_{nullptr};
}；
```

### CpuDeviceContext

CpuDeviceContext is defined as follows:

```c++
class CpuDeviceContext : public DeviceContext{
  Eigen::DefaultDevice eigen_handle() {
    if (!eigen_handle_) {
      eigen_handle_ = new Eigen::DefaultDevice();
    }
    return *eigen_handle_;
  }
private:
  Eigen::DefaultDevice* eigen_handle_{nullptr};  
};
```
