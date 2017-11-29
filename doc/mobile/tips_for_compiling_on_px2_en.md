## Some Information

OS:

	Distributor ID:	Ubuntu
	Description:	Ubuntu 16.04 LTS
	Release:	16.04
	Codename:	xenial

NVCC:

	Cuda compilation tools, release 8.0, V8.0.72

GCC:

	gcc (Ubuntu/Linaro 5.4.0-6ubuntu1~16.04.5) 5.4.0 20160609

CMake:

	cmake version 3.5.1

PaddlePaddle

	commit id: 9876bb0952ad7153d6af44364721fd64e0ce22c7


## Prepare dependency libs

1. Download and compile Boost Library

	```bash
	wget https://dl.bintray.com/boostorg/release/1.65.1/source/boost_1_65_1.tar.gz
	tar zxvf boost_1_65_1.tar.gz && cd boost_1_65_1
	mkdir output && ./b2 install --prefix output
	```

2. Download go

	```bash
	wget https://redirector.gvt1.com/edgedl/go/go1.9.2.linux-amd64.tar.gz
	tar zxvf go1.9.2.linux-amd64.tar.gz
	```

## Adapt cmake flags

* **cmake/configure.cmake patch**

```diff
@@ -69,7 +69,7 @@ else()
         message(FATAL_ERROR "Paddle needs cudnn to compile")
     endif()

-    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler ${SIMD_FLAG}")
+    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "${SIMD_FLAG}")

     # Include cuda and cudnn
     include_directories(${CUDNN_INCLUDE_DIR})
```

* **cmake/flags.cmake patch**

```diff
@@ -99,32 +99,11 @@ SET(CMAKE_EXTRA_INCLUDE_FILES "")
 # Do not care if this flag is support for gcc.
 set(COMMON_FLAGS
     -fPIC
-    -fno-omit-frame-pointer
-    -Wall
-    -Wextra
-    -Werror
-    -Wnon-virtual-dtor
-    -Wdelete-non-virtual-dtor
-    -Wno-unused-parameter
-    -Wno-unused-function
-    -Wno-error=literal-suffix
-    -Wno-error=sign-compare
-    -Wno-error=unused-local-typedefs
-    -Wno-error=parentheses-equality # Warnings in pybind11
 )

 set(GPU_COMMON_FLAGS
-    -fPIC
-    -fno-omit-frame-pointer
-    -Wnon-virtual-dtor
-    -Wdelete-non-virtual-dtor
-    -Wno-unused-parameter
-    -Wno-unused-function
-    -Wno-error=sign-compare
-    -Wno-error=literal-suffix
-    -Wno-error=unused-local-typedefs
-    -Wno-error=unused-function  # Warnings in Numpy Header.
-    -Wno-error=array-bounds # Warnings in Eigen::array
+    -Xcompiler -fPIC
+    -Wno-deprecated-gpu-targets
 )

 if (APPLE)
@@ -134,9 +113,6 @@ if (APPLE)
     endif()
 else()
     set(GPU_COMMON_FLAGS
-        -Wall
-        -Wextra
-        -Werror
         ${GPU_COMMON_FLAGS})
 endif()
```

* **cmake generic.cmake path**

```diff
@@ -94,6 +94,7 @@ if(NOT APPLE AND NOT ANDROID)
     find_package(Threads REQUIRED)
     link_libraries(${CMAKE_THREAD_LIBS_INIT})
     set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl -lrt")
+    set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -pthread -ldl -lrt")
 endif(NOT APPLE AND NOT ANDROID)

 function(merge_static_libs TARGET_NAME)
```

## Adapt \_\_ARM_NEON\_\_ definition macro

* **paddle/function/DepthwiseConvOpTest.cpp patch**

```diff
@@ -34,7 +34,7 @@ TEST(DepthwiseConv, BackwardFilter) {
 }
 #endif

-#if defined(__ARM_NEON__) || defined(__ARM_NEON)
+#if !defined __CUDACC__ && (defined(__ARM_NEON__) || defined(__ARM_NEON))

 TEST(DepthwiseConv, Forward) {
   DepthwiseConvolution<DEVICE_TYPE_CPU, DEVICE_TYPE_CPU>(
```

* **paddle/function/neon/NeonDepthwiseConv.cpp patch**

```diff
@@ -17,7 +17,7 @@ limitations under the License. */

 namespace paddle {

-#if defined(__ARM_NEON__) || defined(__ARM_NEON)
+#if !defined __CUDACC__ && (defined(__ARM_NEON__) || defined(__ARM_NEON))

 template <DeviceType Device>
 class NeonDepthwiseConvFunction : public ConvFunctionBase {
```

* **paddle/function/neon/NeonDepthwiseConv.h patch**

```diff
@@ -20,7 +20,7 @@ limitations under the License. */
 namespace paddle {
 namespace neon {

-#if defined(__ARM_NEON__) || defined(__ARM_NEON)
+#if !defined __CUDACC__ && (defined(__ARM_NEON__) || defined(__ARM_NEON))

 template <int filterSize, int stride>
 struct DepthwiseConvKernel {};
@@ -512,7 +512,7 @@ struct Padding {
   }
 };

-#if defined(__ARM_NEON__) || defined(__ARM_NEON)
+#if !defined __CUDACC__ && (defined(__ARM_NEON__) || defined(__ARM_NEON))
 template <>
 struct Padding<float> {
   static void run(const float* input,
```

* **paddle/function/neon/NeonDepthwiseConvTranspose.cpp patch**

```diff
@@ -17,7 +17,7 @@ limitations under the License. */

 namespace paddle {

-#if defined(__ARM_NEON__) || defined(__ARM_NEON)
+#if !defined __CUDACC__ && (defined(__ARM_NEON__) || defined(__ARM_NEON))

 template <DeviceType Device>
 class NeonDepthwiseConvTransposeFunction : public ConvFunctionBase {
```

* **paddle/function/neon/neon_util.h patch**

```diff
@@ -14,7 +14,7 @@ limitations under the License. */

 #pragma once

-#if defined(__ARM_NEON__) || defined(__ARM_NEON)
+#if !defined __CUDACC__ && (defined(__ARM_NEON__) || defined(__ARM_NEON))

 #include <arm_neon.h>
```

* **paddle/gserver/layers/ExpandConvLayer.cpp patch**

```diff
@@ -94,7 +94,7 @@ bool ExpandConvLayer::init(const LayerMap &layerMap,

     // If depth wise convolution and useGpu == false and ARM-NEON
     if (!useGpu_ && isDepthwiseConv(channels_[i], groups_[i]) && !isDeconv_) {
-#if defined(__ARM_NEON__) || defined(__ARM_NEON)
+#if !defined __CUDACC__ && (defined(__ARM_NEON__) || defined(__ARM_NEON))
       if ((filterSize_[i] == filterSizeY_[i]) &&
           (filterSize_[i] == 3 || filterSize_[i] == 4) &&
           (stride_[i] == strideY_[i]) && (stride_[i] == 1 || stride_[i] == 2)) {
```

* **paddle/math/BaseMatrix.cu patch**

```diff
@@ -667,7 +667,7 @@ void BaseMatrixT<T>::relu(BaseMatrixT& b) {
   applyBinary(binary::Relu<T>(), b);
 }

-#if defined(__ARM_NEON__) || defined(__ARM_NEON)
+#if !defined __CUDACC__ && (defined(__ARM_NEON__) || defined(__ARM_NEON))
 template <>
 void BaseMatrixT<float>::relu(BaseMatrixT& b) {
   neon::relu(data_, b.data_, height_ * width_);
```

* **paddle/math/NEONFunctions.cpp patch**

```diff
@@ -12,7 +12,7 @@ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License. */

-#if defined(__ARM_NEON__) || defined(__ARM_NEON)
+#if !defined __CUDACC__ && (defined(__ARM_NEON__) || defined(__ARM_NEON))

 #include "NEONFunctions.h"
 #include <arm_neon.h>
```

## Disable unittest of DepthwiseConvOpTest

* **paddle/function/CMakeLists.txt patch**

```diff
@@ -50,7 +50,7 @@ endif()

 add_simple_unittest(Im2ColTest)
 add_simple_unittest(GemmConvOpTest)
-add_simple_unittest(DepthwiseConvOpTest)
+#add_simple_unittest(DepthwiseConvOpTest)
 endif()

 add_style_check_target(paddle_function ${h_files})
```

## Make

```shell
mkdir build
cd build && cmake .. -DCMAKE_Go_COMPILER=${GO_BIN} -DBoost_DIR:PATH=${BOOST_PATH}/output -DBoost_INCLUDE_DIR:PATH=${BOOST_PATH}/output/include
make && make install
```

## Compile and install recordio

```shell
git clone https://github.com/PaddlePaddle/recordio.git
cd recordio/python && sh build.sh && pip install recordio-0.1-cp27-cp27mu-linux_aarch64.whl
```

## Install PaddlePaddle wheel

```shell
pip install paddlepaddle-0.10.0-cp27-cp27mu-linux_aarch64.whl
```
