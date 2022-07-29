diff --git a/paddle/fluid/platform/device_context.h b/paddle/fluid/platform/device_context.h
index 2c5f24d28c..ab0d5677d2 100644
--- a/paddle/fluid/platform/device_context.h
+++ b/paddle/fluid/platform/device_context.h
@@ -542,11 +542,11 @@ class CUDADeviceContext : public phi::GPUContext {
   void Wait() const override;
 
   /*! \brief  Return eigen device in the device context. */
-  Eigen::GpuDevice* eigen_device() const;
+  Eigen::GpuDevice* eigen_device() const override;
 
   /*! \brief  Call cublas function safely. */
   inline void CublasCall(
-      const std::function<void(blasHandle_t)>& callback) const {
+      const std::function<void(blasHandle_t)>& callback) const override {
     if (!thread_ctx_.count(this)) {
       phi::GPUContext::CublasCall(callback);
       return;
@@ -557,7 +557,7 @@ class CUDADeviceContext : public phi::GPUContext {
 #ifndef PADDLE_WITH_HIP
   /*! \brief  Call cusparse function safely. */
   inline void CusparseCall(
-      const std::function<void(phi::sparseHandle_t)>& callback) const {
+      const std::function<void(phi::sparseHandle_t)>& callback) const override {
     if (!thread_ctx_.count(this)) {
       phi::GPUContext::CusparseCall(callback);
       return;
@@ -569,7 +569,7 @@ class CUDADeviceContext : public phi::GPUContext {
   /*! \brief  Call cublas function with Tensor Core safely. If
       Tensor Core is not available, use DEFAULT_MATH instead. */
   inline void TensorCoreCublasCallIfAvailable(
-      const std::function<void(blasHandle_t)>& callback) const {
+      const std::function<void(blasHandle_t)>& callback) const override {
     if (!thread_ctx_.count(this)) {
       phi::GPUContext::TensorCoreCublasCallIfAvailable(callback);
       return;
@@ -579,22 +579,22 @@ class CUDADeviceContext : public phi::GPUContext {
 
 /*! \brief  Return cudnn  handle in the device context. */
 #ifdef PADDLE_WITH_HIP
-  miopenHandle_t cudnn_handle() const;
+  miopenHandle_t cudnn_handle() const override;
 #else
-  cudnnHandle_t cudnn_handle() const;
+  cudnnHandle_t cudnn_handle() const override;
 #endif
 
 /*! \brief  Return cublas handle in the device context. */
 #ifdef PADDLE_WITH_HIP
-  rocblas_handle cublas_handle() const;
+  rocblas_handle cublas_handle() const override;
 #else
-  cublasHandle_t cublas_handle() const;
-  cublasLtHandle_t cublaslt_handle() const;
-  cusparseHandle_t cusparse_handle() const;
+  cublasHandle_t cublas_handle() const override;
+  // cublasLtHandle_t cublaslt_handle() const override;
+  cusparseHandle_t cusparse_handle() const override;
 #endif
 
 #ifndef PADDLE_WITH_HIP
-  cusolverDnHandle_t cusolver_dn_handle() const;
+  cusolverDnHandle_t cusolver_dn_handle() const override;
 #endif
 
   /*! \brief  Return a cudnn workspace handle to call multiple cudnn
@@ -604,16 +604,16 @@ class CUDADeviceContext : public phi::GPUContext {
    *  workspace. Once the handle is destructed, the lock would be released.
    *  CudnnWorkspaceHandle is an RAII object to implement thread-safe
    *  sequential cudnn function calls. */
-  phi::DnnWorkspaceHandle cudnn_workspace_handle() const;
+  phi::DnnWorkspaceHandle cudnn_workspace_handle() const override;
 
   /*! \brief  Return cuda stream in the device context. */
-  gpuStream_t stream() const;
+  gpuStream_t stream() const override;
 
-  void RecordEvent(gpuEvent_t ev, const std::function<void()>& callback) const;
+  void RecordEvent(gpuEvent_t ev, const std::function<void()>& callback) const override;
 
-  void AddStreamCallback(const std::function<void()>& callback) const;
+  void AddStreamCallback(const std::function<void()>& callback) const override;
 
-  void WaitStreamCallback() const;
+  void WaitStreamCallback() const override;
 
   void ResetThreadContext(const stream::Priority& priority) {
     std::lock_guard<std::mutex> guard(ctx_mtx_);
diff --git a/paddle/phi/backends/gpu/gpu_context.h b/paddle/phi/backends/gpu/gpu_context.h
index db9f287041..f2d68258bc 100644
--- a/paddle/phi/backends/gpu/gpu_context.h
+++ b/paddle/phi/backends/gpu/gpu_context.h
@@ -88,22 +88,22 @@ class PADDLE_API GPUContext : public DeviceContext {
   const Place& GetPlace() const override;
 
   /*! \brief  Return gpu stream in the device context. */
-  gpuStream_t stream() const;
+  virtual gpuStream_t stream() const;
 
   /*! \brief  Return cudnn  handle in the device context. */
-  dnnHandle_t cudnn_handle() const;
+  virtual dnnHandle_t cudnn_handle() const;
 
   /*! \brief  Return cublas handle in the device context. */
-  blasHandle_t cublas_handle() const;
+  virtual blasHandle_t cublas_handle() const;
 
   /*! \brief  Return cublasLt handle in the device context. */
-  blasLtHandle_t cublaslt_handle() const;
+  virtual blasLtHandle_t cublaslt_handle() const;
 
   /*! \brief  Return cusolver handle in the device context. */
-  solverHandle_t cusolver_dn_handle() const;
+  virtual solverHandle_t cusolver_dn_handle() const;
 
   /*! \brief  Return cusparse handle in the device context. */
-  sparseHandle_t cusparse_handle() const;
+  virtual sparseHandle_t cusparse_handle() const;
 
   /*! \brief  Wait for all operations completion in the stream. */
   void Wait() const override;
@@ -130,7 +130,7 @@ class PADDLE_API GPUContext : public DeviceContext {
   std::array<int, 3> GetCUDAMaxGridDimSize() const;
 
   /*! \brief  Return eigen device in the device context. */
-  Eigen::GpuDevice* eigen_device() const;
+  virtual Eigen::GpuDevice* eigen_device() const;
 
   /*! \brief  Return a cudnn workspace handle to call multiple cudnn
    *  functions without interrupting by other threads.
@@ -139,27 +139,27 @@ class PADDLE_API GPUContext : public DeviceContext {
    *  workspace. Once the handle is destructed, the lock would be released.
    */
   // TODO(wilber): The return type is a pointer, to be modified later.
-  DnnWorkspaceHandle cudnn_workspace_handle() const;
+  virtual DnnWorkspaceHandle cudnn_workspace_handle() const;
 
  public:
   /*! \brief  Call cublas function safely. */
-  void CublasCall(const std::function<void(blasHandle_t)>&) const;
+  virtual void CublasCall(const std::function<void(blasHandle_t)>&) const;
 
   /*! \brief  Call cublas function with Tensor Core safely. If
       Tensor Core is not available, use DEFAULT_MATH instead. */
-  void TensorCoreCublasCallIfAvailable(
+  virtual void TensorCoreCublasCallIfAvailable(
       const std::function<void(blasHandle_t)>&) const;
 
   /*! \brief  Call cusparse function safely. */
-  void CusparseCall(const std::function<void(sparseHandle_t)>&) const;
+  virtual void CusparseCall(const std::function<void(sparseHandle_t)>&) const;
 
-  void RecordEvent(gpuEvent_t ev, const std::function<void()>& callback) const;
+  virtual void RecordEvent(gpuEvent_t ev, const std::function<void()>& callback) const;
 
   void RecordEvent(gpuEvent_t ev) const;
 
-  void AddStreamCallback(const std::function<void()>& callback) const;
+  virtual void AddStreamCallback(const std::function<void()>& callback) const;
 
-  void WaitStreamCallback() const;
+  virtual void WaitStreamCallback() const;
 
  public:
   /*! \brief  Return nccl communicators. */
