# Paddle HIgh reusability operator library (PHI) Design

Paddle HIgh reusability operator library (PHI), or we also call it 'functional operator library', supports to implement new operator kernels based on existing operator kernel functions and 'Kernel Primitives API (KPS)', and supports plug-in access to new hardware or new acceleration library.

In order to solve the problems of unclear operator interface in the original operator library of the Paddle Fluid Framework, high cost of operator reuse, and poor scheduling performance, we refactored the operator library of the Paddle Framework, designed flexible and efficient functional paradigm.

The operator library PHI can implement new operators by combining calls to functional operator interfaces, which can greatly reduce the development cost of native operator and custom operator.

## 1. Background

> Introduce the problems to be solved in designing and building the PHI operator library

The PHI operator library project was initially launched to support the refactoring of the paddle dynamic graph architecture to reduce scheduling overhead and improve the reuse capability of OpKernel development. However, the subsequent decision to take this opportunity to establish an operator library that can be used in both training and inference scenarios (including server-side and mobile-side scenarios), reduce the cost of infrastructure development and operator maintenance in the paddle ecosystem in the long run, so we expanded the target scope of the project.

Specifically, the PHI operator library project carries the expectation to solve the following problems of Paddle.

### 1.1 Poor reusability between Op&OpKernel

Before version 2.3, the reusability between Operators (Op) in Paddle was relatively poor. Only in a few backward Ops, some simple Ops were reused by calling `SetType` in the `GradOpMaker` implementation. In most cases where the existing Op implementation can be reused, the code is rewritten by copy.

The root cause of poor reusability is the inflexibility of the original Op architecture design:

1. When an Op reuses the `Opkernel::Compute` method of another Op, an `ExecutionContext` needs to be constructed first, and the reuse method is relatively cumbersome

    > It will be much more convenient if you can directly call the Kernel in the form of a function

2. Due to the overhead introduced by additional data structure construction and independent Op scheduling, from the perspective of computing performance, it is better to copy the calculation code directly when reusing Op, which leads us to gradually abandon the earlier principle of backward Op reusing forward Op, and began to implement Kernel separately for each backward Op, so that Paddle maintains a large number of backward OpKernel implementation codes internally.

    > Only when the overhead of reusing Ops is small enough, reusing existing Ops to implement new Ops can be widely promoted

### 1.2 Conciseness and fine-grained execution scheduling

#### 1.2.1 Dynamic graph

After the release of Paddle 2.0, it has received many feedbacks from internal and external users that the performance of the dynamic graph is several times lower than that of competing products in the execution scenario of small model on CPU.

The main reason for this problem is: the execution path of the C++ side of the Padddle dynamic graph is relatively long and the scheduling overhead is relatively heavy, which is related to the early design of the dynamic graph which is compatible with the static graph and inherits many object construction processes of the static graph Op.

Therefore, the dynamic graph needs to be upgraded to a function-based scheduling architecture, and this problem can be solved by abandoning the original complex Op architecture, which depends on the OpKernel being changed to a functional writing method.

#### 1.2.2 Static image + IR

Our current static graph mode are not "static" enough. At present, static graph mode still have a lot of logic for dynamic selection at runtime, for example, selecting OpKernel at runtime, judging whether to copy data across devices at runtime, etc.. However, these can actually be determined during the compilation of the static graph mode network, and the execution process is determined as a series of OpKernel executions, and no dynamic judgment selection is made, thereby further improving the execution efficiency.

And these rely on the fine-grained OpKernel itself, decoupling the existing complex large OpKernel into small Kernels for specific scenarios and specific devices.

### 1.3 Ease of use improvement requirements for custom operators

The new custom C++ external operator paradigm released in early 2021 has a relatively intuitive usage at the level of interface and function writing, but because we lack the C++ APIs for basic operations, in fact, when implementing specific custom Op operation logic, such as basic addition, subtraction, multiplication and division and matrix operations, still need to be reimplemented again, and Paddle's existing and optimized basic operations cannot be reused, development costs are still relatively high. In order to reuse the basic operations inside Paddle, the Op paradigm must be upgraded to functional paradigm, and build the corresponding C++ API system.

### 1.4 Build an integrated training and inference operator library to reduce the maintenance cost of inference operators

For a long time, because the Paddle and Paddle-Lite operators are maintained separately, the new paddle operator, if Paddle-Lite needs it, must be manually reimplemented in Paddle-Lite, and when the Paddle operator is upgraded, Paddle-Lite does not perceive it in time, which will directly lead to bugs in the inference model when lite is executed, which introduces high maintenance costs. Only a unified operator library can solve this problem for a long time.

Therefore, this functional operator library will be jointly constructed by training and inference team, and will serve as an independent compilation component and underlying infrastructure (not yet independently split), which can serve training, server-inference, and -inference execution systems at the same time.

### 1.5 The adaptation of the new inference Runtime design 'infrt'

Inference team designed a new runtime `infrt`. It is expected to unify the execution system of Paddle-Inference and Paddle-Lite. It is necessary to directly call the operators in the PHI operator library jointly built this time. Therefore, the adaptation to `infrt` needs to be considered in the design. (Currently the `infrt` project is temporarily on hold).

### 1.6 Op and Kernel parameter normalization

The Python 2.0 API project in 2020 standardized the argument list of the Paddle Python-side API, making it concise, easy to use, and standard. However, due to cost considerations, the argument list at the Op level was not standardized, so there will be many early developed operators that differ greatly in arguments from the Python API. For example, `conv` op, the Python API has only 8 arguments, but the corresponding C++ `Conv` Op has 29 arguments. API and Op are essentially the same layer of concepts, both are descriptions of an operation, and the arguments should be consistent. In order to solve this problem, 'the operator definition enhancement project' was launched, and the declarations of 'AsExtra' and 'AsQuant' were added to some unnecessary arguments, but the problem was not fundamentally solved, which is what the construction of the PHI operator library hopes to solve.

We hope to be able to achieve the same three-layer arguments of Python API -> Op(C++ API) -> Kernel API, so that the overall structure is clear, and the reuse relationship of each layer is clear enough. Maintaining a set of official Python API documents can basically satisfy the common reference requirements of the three-tier API, no longer focus on maintaining additional document systems and reduce maintenance costs.

## 2. Design Overview

### 2.1 Location

The PHI code directory is inside the paddle directory, which is at the same level as fluid, rather than inside the fluid directory. PHI is a basic component that is called by various upper-layer runtime such as fluid, lite, and infrt, and it will be used later as a separately compiled dynamic library, therefore PHI is not suitable as the submodule of fluid.

### 2.2 Directory Structure

#### 2.2.1 Requirements of directory structure design

Training and inference require a clear operator library directory structure:

- The directory design should support various split compilation requirements of the operator library, which including:

    - Split and compile according to the computing device.
        - For example, compile for cpu only, or compile for gpu only.
    - Split and compile according to the training and inference scenarios.
        - For example, the inference scenario does not compile backward-relevant kernels (xxx_grad_kernel.cc|cu)
    - Precisely crop and compile according to the operators actually used by the mobile device (not supported yet)
        - For example, a model uses `add` and `multiply` only, ideally it could be cropped to only 2 kernels.

- In the long run, support the requirement of easily reusing kernel implementation.
    - Explanation: When reusing the kernel, the corresponding function implementation should be introduced through `include` easily, rather than cannot find the kernel because of the complex directory structure.

- In the long run, support the requirement of the unified writing method among cross-device kernels, and the writing method is intuitive and easy to use, without introducing unnecessary template parameters.
    - Explanation: Kernel Primitive API module is at the lower layer of the operator library. Its long-term vision is that each operation uses only one kernel to adapt to various devices, the code that truly distinguishes the device is only in the implementation of the Kernel Primitive API. In the future, the template parameters should be limited to as concise as possible when passing complex parameters into the reused kernel.

- In terms of ease of use, developers can accurately understand where the newly added kernel should be placed, without ambiguity.
    - Explanation: When developers add an API, they will not be confused about which directory they should put the corresponding kernel in. Moreover, different people should have no ambiguous understanding of where the same kernel should be placed.

- Do not introduce a lot of duplicate directory design.
    - Explanation: Concept splitting is needed, but also with boundaries. Avoid subdirectories with the same name occurring in multiple directories. For example, if `eigen`, `funcs`, `math` directories are placed under the cpu directory, then they shouldn't be placed under the gpu directory. The directory design of the new operator library is mainly divided according to the device, and the directory splitting at other levels should be weakened as much as possible. For example, try not to split based on functions, try not to split based on fields, etc.

- Do not introduce too deep directory design.
    - Explanation: The directory level should not be too deep, otherwise it will lead to higher understanding and maintenance costs.

#### 2.2.2 Directory design details

##### 2.2.2.1 First level directory

```
paddle/phi
./api (High-level API exposed to the outside and corresponding implementation)
    ./include (High-level API header file exposed to the outside)
    ./lib (API implementation exposed to the outside)
    ./yaml (Operator Definition in yaml Configuration Form)
./capi (C API exposed to the outside and corresponding implementation)
    ./include
    ./lib
./common (Basic data structures used both internally and externally)
./core (Basic components, such as basic Tensor-related interfaces, kernel registration interfaces, management units, etc.)
./backends (Basic components of each device and backend, including backend directories such as cpu, gpu, etc.)
./infermeta (Derivation functions for meta information such as shape, dtype, layout, etc.)
./kernels (Kernel implementation of each device and backend)
./ops (Operator implementation related code, only contains code for compatibility with fluid)
./tests (Unit test)
```

Some directory structure description:

- `api`: API module for external users. Directly use the Python-like C++ Tensor computing API, which is highly consistent with the Python side.
- `capi`: C API module, mainly serving the plug-in hardware access function.
- `common`: Data structures to be used both inside PHI `core` and PHI `api` directory. These data structures neither belong to the `core` nor the `api` directory.
- `core`: PHI has some public module implementations that it needs, such as `DenseTensor`, kernel registration and management modules.
- `backends`: The backends include data structures that need to be added for each backend, such as `CPUContext`, `GPUContext`, etc.
    - The basic data structures are placed in the `core`, while the dedicated data structures of specific backends are not placed in the `core`, and the dependencies strictly ensure that the `backends` depend on the `core`, but the `core` cannot depend on the `backends`.
    - Example 1: If Context is a base class, then put it in `core`, inherited `CPUContext` is in `backends/cpu` and `GPUContext` is in `backends/gpu`.
    - Example 2: TensorBase is in `core`, `DenseTensor` is used by most devices so that it is also in the `core`. If there is `OneDNNTensor`, which is only used for `OneDNN`, then it should be placed in `backends/onednn`.
- `infermeta`: The location of the infermeta function, the infermeta function is equivalent to `infershape + inferdtype + inferlayout`, etc.
- `kernels`: Kernels related to each device.
    - `cpu, gpu, ...`


##### 2.2.2.2 Kernels directory

```
paddle/phi/kernels
./ (Device-independent kernel declarations and implementations.)
./cpu (Include the kernel implementation of the cpu backend only.)
./gpu
./xpu
./onednn
./gpudnn
./impl (A special temporary directory that contains the kernel implementation used by the CPU and GPU only)
./funcs (Including some functor and function that support multiple devices under the original fluid operators)
./primitive (Includes basic implementation of the Kernel Primitive API)
...
```

The directory structure is described as follows:

- The root directory under kernels includes device-independent `kernel.h` and `kernel.cc`. In principle, each kernel has one .h and .cc
    - For example, if a kernel is implemented using Primitive api, or is implemented by reusing other basic kernels, there should be only one implementation for all devices, so its declaration and implementation can be placed directly in the kernels directory. (This is the ideal state in the future.)
    - At present, most of our kernels do not have the feature of unity implementation across devices, but the input parameters and return values of the kernel should be consistent except for `DeviceContext`, so the kernel parameter declaration header file is also placed in the current directory (consistent with the original design, `DeviceContext` and `T` are used as template parameters), The functions implementation of each device are placed in the corresponding device folder.
        - Note that the unity implementation across devices here does not mean that the CPU and GPU implementations of a kernel are unified, but the implementations of all devices are the same. Currently, it includes at least `CPU`, `GPU`, `XPU`, `ONEDNN`, `GPUDNN`, etc.
    - If the backward kernel does not need to support cropping, it can be merged appropriately (but if you want to leave the possibility of supporting end-to-side training, the backward kernel may also be a potential target for cropping)
- The next-level subdirectory of kernels, in principle, is created according to the backend classification, and only two special directories are reserved:
    - `funcs`: In order to be compatible with the directories of functor and function in the original fluid/operators directory, when placing functions and functor that support multiple backends, we organize them according to the original design that one header file corresponding to multiple .cc(u) (This part of the code may be removed in the future, because it will be gradually replaced by Kernel Primitive API and reuse between Kernels, so no over-design here.)
        - Example 1: A common function `XXXFunction` is called in both reduce CPU and reduce GPU kernel implementations, and the reduce CPU and reduce GPU kernel implementations are different, then `XXXFunction` should be in the `funcs` directory.
    - `primitive`: Kernel Primitive API, some basic tools for multi-device unified kernel implementation.
    - `impl`: Many paddle's original op kernel implementation reuse the same code for CPU and GPU, and they are in a large number of `xx_op.h`. This part of the code is not suitable to be placed in the `cpu` or `gpu` directory, nor in the `funcs` directory (putting it in the `funcs` directory will cause a considerable part of the kernel implementation to be placed in the `funcs` directory, which is too bloated and confusing. The `funcs` directory is created to place the `functor` and `function` tools as in the original operators/math directory). This part of the code is also not suitable to be placed in the root directory of `kernels` (it is not a device-independent implementation, only an implementation shared by cpu and gpu). Therefore, in order not to overthink this part of the code when migrating, and the location of the placement is relatively consistent with its implementation nature, the `impl` directory was created.
        - In the `impl` directory, only the kernel functions that are consistent across some devices are placed. They are all header files, and the names are all suffixed with `xxx_kernel_impl.h`
        - For example: `scale`, `fill_constant`, `fill_any_like` kernels are all such cases.
- The auxiliary functions that are only used by the current kernel, they are always placed in the same backend folder as the kernel implementation, and the .h file is used to manage the code. Auxiliary function codes are no longer placed elsewhere, unless their implementations are used in multiple places.
    - Even if there are multiple calls, if it is still limited to the same device, directly build the header file and put it in the same directory.
- The implementation of the backward kernel and the forward kernel are placed in different files, and the file suffix is `*_grad_kernel.*`, which is convenient for cmake to separate and compile.
    - No more directories are created for the backward kernel, otherwise directories such as cpu/gpu will also be created under the backward kernel directory.
    - The implementation of the second-order derivative and the third-order derivative is also placed in the grad kernel implementation file.

- Why is the directory named `gpu` instead of `cuda` and `hip`?
    - The code of `cuda` and `hip` is very repetitive, and the unified implementation is easier to maintain.

#### 2.2.3 Namespace

PHI is an API-oriented operator library, so it does not fully follow the code-style here. Data structures and functions in `core` and `kernels` may be directly exposed to ordinary users in the future, so we hope that the namespace prefix of functions can be shorter so that easy to call. The design principles here are as follows:

1. API-level classes and functions are placed under the `namespace phi`.
2. The namespace where non-API-level class implementations, auxiliary tool implementations, and auxiliary function implementations are still based on the relative path of the code as much as possible.

### 2.3 Core Components

#### 2.3.1 Common basic data structure

##### 2.3.1.1 Backend、DataLayout And DataType

See the source code for detailed design:

- Backend: `paddle/phi/common/backend.h`
- DataLayout: `paddle/phi/common/layout.h`
- DataType: `paddle/phi/common/data_type.h`

##### 2.3.1.2 Scalar

Scalar is used to uniformly represent variables with different basic data types (float, double, int, bool, etc.). (Currently, Tensor scalars representing 1 element are also supported, but support for this feature may be dropped in the future)

Take `ScaleKernel` as an example, the `scale` parameter can be passed in int, float, double and other basic data types. If you do not use `Scalar` to represent, you need to create a separate functional interface for each data type, which will greatly increase the amount of code when developing Kernel, so `Scalar` is mainly applied to the parameter with different data types, which avoids implementing multiple overloaded functions.

```c++
template <typename T, typename Context>
void ScaleKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const Scalar& scale,
                 float bias,
                 bool bias_after_scale,
                 DenseTensor* out);
```

##### 2.3.1.3 IntArray

IntArray is an integer type array that can be constructed from `vector<int>`, `Tensor` and `vector<Tensor>`. Currently, it is mainly used to represent dimension index variables such as `shape`, `index` and `aixs`.

Taking `FullKernel` as an example, the shape parameter is used to indicate the dimension information of the returned Tensor (e.g. [2, 8, 8]). When calling `FullKernel`, the parameters of `vector<int>`, `Tensor` and `vector<Tensor>` type variables can be used to complete the call. Using `IntArray` avoids the problem of writing a separate overloaded function for each shape type.

```c++
template <typename T, typename Context>
void FullKernel(const Context& dev_ctx,
                const IntArray& shape,
                const Scalar& val,
                DenseTensor* out);
```

#### 2.3.2 Tensor Design

##### 2.3.2.1 API Tensor interface

- The top-layer is the API-level Tensor interface, which contains two pointer members, `TensorBase` and `AbstractAutogradMeta`.
    - Both members are designed as Interface and do not depend on real Tensor and `Autograd` implementations.
    - `AutogradMeta` is only meaningful in the dynamic graph API-level Tensor, it will not be used in the specific kernel calculation, so put it in the top-layer Tensor interface.
    - In addition, such a design facilitates data sharing and reduces copy overhead.
        - When a Tensor is assigned to another Tensor, or Tensor is used as a function return value, only the pointer is actually copied, and no real data copy is performed.

- The top-layer C++ Tensor plays a similar role as the Python-side Tensor, and the interface design is as consistent as possible with the Python-side.
    - Contain basic property access and data access methods of Tensor.
        - `shape`, `place`, `dtype`, `data`.
    - Contain the `autograd` methods required by the dynamic graph Tensor.
        - `gradient`, `backward`.
    - Contain conversion methods between Tensors.
        - cpu, gpu, xpu etc.
    - Contain calculation methods related to Tensor (not added yet).
        - All methods of the `paddle.tensor` module.

- Compilation decoupling:

    - The `autograd` information here is just a pointer index, which is empty by default.
        - `std::unique_ptr<AbstractAutogradMeta> autograd_meta_ = nullptr;`
    - `AbstractAutogradMeta` is an abstract class interface that does not depend on any module of `autograd`, so it will not affect the independent compilation of PHI, and at the same time takes into account the need for dynamic graph Tensor to hold backward information.

- `AutogradMeta` is only set in the dynamic graph scenario. For unneeded scenarios, such as in static graphs, `AutogradMeta` is just a null pointer.

Devices judgment and conversion of Tensor.

- The judgement method of Tensor device and type.

```c++
bool is_cpu() const;
bool is_gpu() const;
bool is_xpu() const;
bool is_dense_tensor() const;
bool is_selected_rows() const;
bool is_opencl() const; // To be added
bool is_metal() const;  // To be added
```

- The type conversion method between Tensors, which is implemented through the API consistent with the Python side (to be added).

```c++
Tensor cpu() const; // Convert to cpu tensor
Tensor gpu() const; // Convert to gpu tensor
Tensor xpu() const;
Tensor ondnn() const;
```

- This conversion process may be `cast` or `copy`:
    - `cast` if no data copy required.
    - `copy` if data copy required.
    - Transformations are implemented by functional kernels.

- Usage in API Scenarios
    - In a complete training scenario, when a user uses an API, such as `DataLoader`, the data is generally read from the disk, put it into the CPU, and then converted to the specific execution device.

##### 2.3.2.2 TensorBase

- The interface implemented by Tensor only contains the necessary pure virtual Tensor methods, and does not contain members with real meaning. The methods here should also be strictly monitored during the development process.

- Why use abstract class design at this level?
    - On the one hand, it is to isolate the Tensor API from the specific implementation of Tensor without generating too many dependencies. If the Tensor API needs to be redesigned in the future, or the `autograd` information needs to be abandoned, only the Tensor API needs to be redesigned, which has little effect on the implementation of the underlying Tensor.
    - On the other hand, in order to reserve sufficient expansion space for heterogeneous Tensors, the framework-level API only needs one Tensor data structure, and there is no need to expose multiple data structures. In fact, a large-scale definition is made here: all data structures in the framework are Tensors.
        - For a basically consistent memory layout, or a basically consistent implementation of Tensor descriptions, it can be inherited based on an implementation of `DenseTensor`.
        - For Tensors with a high degree of heterogeneity, new Tensor classes (such as Tensors with only one Object) can be directly inherited from Interface. This ensures that Tensor has no bottlenecks in scaling flexibility.

##### 2.3.3.3 DenseTensor、SparseTensor

- `DenseTensor` is the basic implementation of Tensor, which corresponds to the `LoDTensor` class in the original fluid. `DenseTensorMeta` in `DenseTensor` contains basic members that describe Tensor information, and `Allocation` in `DenseTensor` is the original `Allocation` of fluid.
- `SparseCsrTensor` and `SparseCooTensor` are newly designed sparse Tensor types, see code implementation for details.

> In order to be compatible with the original framework scheduling and operators, we have also migrated `SelectedRows` as a basic Tensor type. If it can be replaced by a new sparse Tensor in the future, it will be removed.

##### 2.3.3.4 Other Heterogeneous Tensors

- If the existing `Allocation` cannot meet the Tensor memory requirements of some third-party libraries, you can use the new `Allocation` implementation after inheriting `TensorBase`.
- This kind of Tensor is not essentially out of the scope of general Tensor, but the memory access method is different, it still needs other `TensorMeta` information.
- To build a custom Tensor, you can define a special `TensorAllocation` description class, such as `MetalTensor`.

```c++
template <typename AllocationType>
class SpatialTensor : public TensorBase {
 public:
  SpatialTensor(std::shared_ptr<AllocationType> allocation,
                std::unique_ptr<DenseTensorMeta> meta)
      : allocation_(std::move(allocation)),
        meta_(std::move(meta)) {}

 private:
  std::shared_ptr<AllocationType> allocation_;
  std::unique_ptr<TensorMeta> meta_;
};

template <typename AllocationType>
class MetalTensor : public SpatialTensor<AllocationType> {};

template <typename AllocationType>
class OpenCLTensor : public SpatialTensor<AllocationType> {};
```

- In this way, no matter how special the needs of Tensor are, it can be internally adapted on the premise of keeping the external API consistent.

Inherit other Tensors with high degrees of freedom: directly inherit `TensorBase`.

- `TensorBase` is an abstract class, which leaves a lot of room for the description of specific Tensor. If the description of traditional Tensor cannot meet the requirements, a specialized Tensor implementation can be designed.


#### 2.3.3 C++ API

##### 2.3.3.1 C++ API form

> Highlights of this section:
> 1. The C++ API corresponds to the Python 2.0 API: the function name, parameter name, parameter order, and return value are the same.

After investigation, we found that very few framework products are designed with the ease of use of the C++ API in mind. For the long-term consideration, if we want to attract more developers to build the paddle ecology, it is also very important to provide a standardized and easy-to-use C++ API architecture. At the same time, the Python 2.0 API project has laid a good reference foundation for the C++ API, and we can directly inherit its achievements.

Therefore, currently we expect the C++ API declaration form of the Tensor computing library to be as follows:

```c++
Tensor mean(const Tensor& x);

Tensor scale(const Tensor& x,
             const Scalar& scale,
             float bias,
             bool bias_after_scale);
```

Described as follows:

- It should be as consistent as possible with the attributes of the Python API, and the function name, parameter list, and return value should be consistent, so that users will not have any additional learning costs in the switch between Python and C++ (if they must be inconsistent, new C++ APIs can be added. There is a one-to-one correspondence between Python's existing APIs and C++ APIs).

**What scenarios is this new C++ API architecture mainly used for?**

1. C++ API that can be called when developing custom operators, it improves ease of use.
    - For example, the user needs to initialize a Tensor in a custom operator, loop through the Tensor data and assign values, then you can directly call `paddle::ones`, `paddle::full` APIs.
2. The architecture serves as the basic calling unit of the new dynamic graph.
    - The new dynamic graph will use the API as the scheduling calculation unit, and will no longer call the Op architecture, thus improving the scheduling performance.
3. As a basis for the development of backward Op reuse forward Op.
    - Now the backward op kernel needs to be implemented separately. After the API architecture is completed, it is hoped that the backward op implementation can be completed by reusing the forward API.

##### 2.3.3.2 C++ API auto-generate

**Why auto-generate C++ API?**

- The implementation code of the C++ API is relatively fixed in form, and can theoretically be implemented in an auto-generate way.
- Using automatic code generation can effectively reduce the development cost of C++ API, and it is easy to modify and maintain.

**How to automatically generate C++ API?**

The automatic generation of the C++ API is generated by parsing the YAML configuration file. The YAML configuration file is divided into:

- Forward API configuration file(`paddle/phi/api/yaml/api.yaml`. After parsing, the generated code file is `paddle/phi/api/include/api.h` and `paddle/phi/api/lib/api.cc`)
- Backward API configuration file(`paddle/phi/api/yaml/backward.yaml`. After parsing, the generated code file is `paddle/phi/api/backward/backward_api.h` and `paddle/phi/api/lib/backward_api.cc`)

The key to C++ API generation lies in the configuration of the YAML file. Taking `matmul` as an example, the forward and backward configuration are as follows:

```yaml
## Forward API configuration
- api : matmul
  args : (Tensor x, Tensor y, bool transpose_x=false, bool transpose_y=false)
  output : Tensor
  infer_meta :
    func : MatmulInferMeta
  kernel :
    func : matmul
  backward : matmul_grad

## Backward API configuration
- backward_api : matmul_grad
  forward : matmul (Tensor x, Tensor y, bool transpose_x, bool transpose_y) -> Tensor(out)
  args : (Tensor x, Tensor y, Tensor out_grad, bool transpose_x=false, bool transpose_y=false)
  output : Tensor(x_grad), Tensor(y_grad)
  infer_meta :
    func : MatmulGradInferMeta
  kernel :
    func : matmul_grad
```

The meaning of each configuration parameter:

- api: function name, which must be the same as the function name registered by PHI Kernel.
- args: the function parameters. Their order and data type must be exactly the same as the PHI Kernel function of the same name, and the `Attributes` type must be ranked after the `Tensor` type.
- output: the output type. If there are multiple outputs, then separate them by commas (","). You can optionally mark the name of each input with "()" after the type (e.g. `Tensor(out)`). If there is no mark, the default markers is `out0`, `out1`, ...
- infer_meta: calculate the dimension and type of the returned Tensor (see the introduction of the `InferMeta` function for details).
    - func: the called `InferMeta` function. It's default input is all the parameters of the args item and the output parameter of api, the Tensor type variable in it will be automatically replaced with `MetaTensor`.
- kernel: the specific Kernel function called by the API.
    - func: the registered name of the kernel function (the name used by `REGISTER`, not the function name). It's default input is all the parameters of the args item and the output parameter of api.
- backward: (optional). The corresponding backward function name, if not set only the forward API will be generated.

The YAML parsing script will automatically generate the corresponding C++ API according to the above configuration items. The generated code includes the relevant processing logic such as Kernel automatic selection, Tensor transformation, Data Transform, `InferMeta` and Kernel calling. For details, please refer to the generated code in `api.cc` .

Due to the large number of C++ APIs and their various forms and functions, some more flexible configuration items are also provided in the YAML configuration mechanism, such as `invoke`, etc. In the future, it is expected that some new configuration items will be added as needed.

#### 2.3.4 Kernel Form, Registration and Management

##### 2.3.4.1 Kernel form

> Highlights of this section:
> 1. Notes on Kernel function form:
> (1) Data type `T` and `DeviceContext` (abbreviated as `Context`) as template parameters;
> (2) `Context` is the first parameter of Kernel;
> (3) The return value Tensor takes the form of a pointer as an input parameter, and the return value of Kernel itself is void.

This part includes the specific Kernel. The functions implemented in this part will be registered in the framework as Kernel for unified search and scheduling by the framework.

Currently we expect this part to be of the following form, using `scale` as an example:

```c++
template <typename T, typename Context>
void Scale(const Context& dev_ctx,
           const DenseTensor& x,
           float scale,
           float bias,
           bool bias_after_scale,
           DenseTensor* out) {
  ...
}
```

Described as follows:

- The kernels of different devices must have different function implementations. The function names are named in **camel case**. Except for the capitalization of the first letter, the naming should be as consistent as possible with the API function name. The function names of the same calculation are kept the same, and the functions of different devices are managed through different files or directories.
- There are generally two template parameters, `T` and `Context`, which are used to determine the data type and device type at runtime.
    - According to our current architecture, the vast majority of Kernels reduce the code in the way of **specialized DeviceContext and data type**, which is consistent with the original `OpKernel` form.
    - The form should be unified. If the Kernel level is also exposed as a fine-grained API in the future, the ease of use is guaranteed.
- Specification of function input parameters:
    - Take a specific `DeviceContext` (such as `CPUContext`, `GPUContext`) as the first input parameter to meet the needs of specific context information required at runtime. Pass the stream in if there are multiple streams.
        - Currently, it is not supported to pass multiple `DeviceContext` parameters to one Kernel. At present, such a requirement is considered unreasonable.
    - The parameter list is consistent with the API. If there is other special information that needs to be passed into the Kernel, pass it through the `Context`.
    - Then all input Tensors and input Attributes are passed in with const &, and POD types are passed in directly by value.
    - The input Tensor is a specific Tensor type, such as `DenseTensor` or `SelectedRows`, not the Tensor of the external interface API.
    - Finally, the Tensor return value of the function, passed in as a pointer.
    - In order to make the mechanism more flexible and allow the kernel to adapt to more scenarios, the declaration of flexible types of input, output and parameters will be allowed subsequently to adapt to non-Tensor input, output and Tensor Attribute.
- The internal implementation of the function is determined on demand:
    - Short term:
        - Migrate the implementation of the existing `OpKernel` to the specific device Kernel.
        - Abstract the implementation of `OpKernel` with common devices into functions, which are called by multiple device Kernels.
    - Long term:
        - The complex kernel directly calls the basic kernel to complete the calculation, encourages kernel reuse, thus simplifies the code.

> FAQ:

>- Why does the first parameter need to be `DeviceContext`? Why must this parameter be passed in?
    - The PHI kernel requires a pure function format. The variables used in the function are passed in through parameters or created inside the function, global singletons are not allowed inside the function. In order to adapt to various kernel requirements, the `DeviceContext` parameter that stores context information is necessary.
>- Why are two template parameters needed?
    - In order to efficiently support the reusing of device-independent kernels. If we want to implement a Fourier transform `fft` kernel, assuming that the kernel can be derived by combining the basic kernels, the form of `Xxx<T, Device>()` can avoid dynamically redistributing devices.

##### 2.3.4.3 Kernel implementation

> Highlights of this section:
> 1. Kernel focuses on computing logic without mixing scheduling logic.
> 2. Kernel is fine-grained enough, with clear boundaries, no optional parameters, easy to reuse.

The existing Kernel introduces scheduling logic because the Op parameter is too complex, for example:

- Use `use_cudnn` to determine whether to execute the cudnn branch. In the new Tensor calculation library, the use of `cudnn` calculation is a separate Kernel.

In order to reduce costs, the PHI Kernel implementation will inherit the original `OpKernel` implementation as much as possible. Most Kernel implementations only need to remove the `Input` and `Output` logic from the original `OpKernel` and modify some key points. Take `sign` as an example:

Original `sign` OpKernel:

```c++
template <typename DeviceContext, typename T>
class SignKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& context) const {
    auto* out = context.Output<framework::Tensor>("Out");
    auto* in = context.Input<framework::Tensor>("X");
    out->mutable_data<T>(in->place());

    auto eigen_out = framework::EigenVector<T>::Flatten(*out);
    auto eigen_in = framework::EigenVector<T>::Flatten(*in);
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    EigenSign<std::decay_t<decltype(place)>, T>::Eval(place, eigen_out,
                                                      eigen_in);
  }
};
```

Migrated PHI sign kernel:

```c++
template <typename T, typename Context>
void SignKernel(const Context& dev_ctx,
                const DenseTensor& x,
                DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto eigen_out = EigenVector<T>::Flatten(*out);
  auto eigen_x = EigenVector<T>::Flatten(x);

  auto& dev = *dev_ctx.eigen_device();
  funcs::EigenSign<std::decay_t<decltype(dev)>, T>::Eval(
      dev, eigen_out, eigen_x);
}
```

In addition to the change of kernel form from structure format to functional format, there are two major changes:

1. Since the parameters are all specific inputs, there is no need to take the input and output from the context, the relevant code is removed.
2. In the PHI kernel, the memory application of the output Tensor is required to use the `ctx.Alloc` or `ctx.HostAlloc` method, and no longer use the original `mutable_data` to apply for memory.

> FAQ
> 1. Why is `mutable_data` replaced by `ctx.Alloc`?
> Answer: Because the global method `memory::AllocShared` called in the original `mutable_data` method uses a global singleton for memory allocation, which does not conform to the pure function design principle mentioned above. In terms of business requirements, if a single instance is used in the kernel to determine the way of memory allocation, in the multi-threaded environment of inference, different threads will not be able to flexibly specify different memory allocation ways.


##### 2.3.4.4 Kernel registration

> Highlights of this section:
> 1. Kernel needs to expose all its key information to the framework and record its input, output and attribute information, otherwise it will lead to unclear boundaries between framework scheduling and Kernel calculation.

When fluid Kernel is registered, only the `place`, `layout`, `dtype`, `input` and `output` of the Kernel are recorded and managed by `ExecutionContext`, and there is no corresponding information record. Now the kernel needs to be changed to a functional type. The input, output and attributes of each function are clear. We hope to record the information of each input and output here, which is also compatible with paddle-lite scheduling.

Meanwhile, we need to simplify the writing method of Kernel registration. The existing writing methods are not concise enough:

1. There is a lot of redundant information in the Kernel registration method of fluid. Taking `scale` as an example, you can see that in addition to the last data type of each kernel, the preceding function names and `DeviceContext` specialization information are redundant.

    ```c++
    REGISTER_OP_CPU_KERNEL(
        scale, ops::ScaleKernel<paddle::platform::CPUDeviceContext, float>,
        ops::ScaleKernel<paddle::platform::CPUDeviceContext, double>,
        ops::ScaleKernel<paddle::platform::CPUDeviceContext,
                         paddle::platform::bfloat16>,
        ops::ScaleKernel<paddle::platform::CPUDeviceContext, uint8_t>,
        ops::ScaleKernel<paddle::platform::CPUDeviceContext, int8_t>,
        ops::ScaleKernel<paddle::platform::CPUDeviceContext, int16_t>,
        ops::ScaleKernel<paddle::platform::CPUDeviceContext, int>,
        ops::ScaleKernel<paddle::platform::CPUDeviceContext, int64_t>);
    ```

2. Paddle-Lite's kernel registration method declares input and output information for each Kernel, but since the kernel of each data type is different, it will also cause redundancy in the writing method. As you can see in the following code, except for the data type, other information is basically redundant.

    ```c++
    #ifdef LITE_BUILD_EXTRA
    using scale_int32_f =
        paddle::lite::kernels::arm::ScaleCompute<int, PRECISION(kFloat)>;
    REGISTER_LITE_KERNEL(scale, kARM, kFloat, kNCHW, scale_int32_f, int32)
        .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
        .Finalize();

    using scale_int64_f =
        paddle::lite::kernels::arm::ScaleCompute<int64_t, PRECISION(kFloat)>;
    REGISTER_LITE_KERNEL(scale, kARM, kFloat, kNCHW, scale_int64_f, int64)
        .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
        .Finalize();
    #endif  // LITE_BUILD_EXTRA

    #ifdef ENABLE_ARM_FP16
    using scale_float16 =
        paddle::lite::kernels::arm::ScaleCompute<float16_t, PRECISION(kFP16)>;
    REGISTER_LITE_KERNEL(scale, kARM, kFP16, kNCHW, scale_float16, def)
        .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
        .Finalize();

    #endif  // ENABLE_ARM_FP16

    using scale_float =
        paddle::lite::kernels::arm::ScaleCompute<float, PRECISION(kFloat)>;
    REGISTER_LITE_KERNEL(scale, kARM, kFloat, kNCHW, scale_float, def)
        .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
        .Finalize();

    using scale_int32 =
        paddle::lite::kernels::arm::ScaleCompute<int, PRECISION(kInt32)>;
    REGISTER_LITE_KERNEL(scale, kARM, kInt32, kNCHW, scale_int32, def)
        .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
        .Finalize();

    using scale_int64 =
        paddle::lite::kernels::arm::ScaleCompute<int64_t, PRECISION(kInt64)>;
    REGISTER_LITE_KERNEL(scale, kARM, kInt64, kNCHW, scale_int64, def)
        .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
        .Finalize();
    ```

Therefore, in this design, we do not want to continue to maintain this redundant writing method. We hope that the writing method of kernel registration is concise enough, and at the same time, it can flexibly meet the requirements of Kernel input and output information configuration.

The key point of this problem is that the kernel needs to specify its own `device`, `layout` and `dtype` as its own key information, and most of the kernel input and output Tensor's `device`, `layout` and `dtype` are consistent with the kernel itself. For this kind of kernel, there is no need to declare through `BindInput` and `BindOutput`, we can automatically generate information to fill each input and output according to the information of the kernel. We only need to configure special information for the input and output that is inconsistent with the kernel information.

The new kernel registration form is as follows:

```c++
PT_REGISTER_KERNEL("sign", CPU, NCHW, pt::Sign, float, double) {}

PT_REGISTER_KERNEL("mean", CPU, NCHW, pt::Mean, float, double) {}

PT_REGISTER_KERNEL("scale", CPU, NCHW, pt::Scale, float, double, bfloat16,
                   uint8_t, int8_t, int16_t, int, int64_t) {}

PT_REGISTER_KERNEL("scale_host", CPU, NCHW, pt::ScaleHost, float, double, bfloat16,
                   uint8_t, int8_t, int16_t, int, int64_t) {
   kernel->InputAt(1).SetBackend(pt::Backend::kCPU);
}
```

Described as follows:

- A large amount of redundant information in the previous registration method is removed, and the scale kernel registration of 8 data types can be completed with one line of code, and the information of each input and output is recorded by default according to the kernel information.
- For a kernel with dynamic attribute input such as `ScaleTensor`, you can configure the `Backend`, `Layout` and `Dtype` information of specific parameters in the function body; if there is no such requirement, the function body can be empty.

In addition, in the `PT_REGISTER_KERNEL` macro, the function form of the Kernel function is normalized through template deduction.

The kernels with different input parameter lists are unified into the following form, so that they can be stored in the Kernel data structure below as a unified function pointer:

```c++
using KernelFn = void (*)(KernelContext* ctx);
```

Auto-derivation by wrapping `PT_KERNEL` around the Kernel function

```c++
##define PT_KERNEL(...) \
  ::pt::KernelImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::Compute
```

In addition, only basic template adaptation has been implemented at present, and we will add them as needed in the future to make the overall mechanism more flexible and applicable to a wider range.

##### 2.3.4.4 Kernel management

> Highlights of this section:
> 1. Introduce the design of the current Kernel management components

For the management of the new form of Kernel, described as follows:

- `KernelFactory` is a global singleton data structure for managing Kernel. Similar to `OpKernelMap` of fluid, it is a two-level map. The first-level mapping finds the Kernel set according to the name, and the second-level mapping finds the specific Kernel according to the KernelKey.
- `KernelKey` is similar to the original `OpKernelType`, but the `palce` and `library_type` fields are combined into one and called `Backend`, because the original `LibraryType` is a limited enumeration class, which is strongly related to place, the splitting increases the cost of understanding instead.
- `Kernel` holds more information than the original `OpKernel`. In addition to the Function during execution, it also holds information about specific parameters, namely `KernelArgsDef`. For Tensor type input and output, it saves Tensor type information, Device, data Type, data layout. For Attribute type input and output, it saves type information.


#### 2.3.5 Kernel Compilation and Dependencies

> Highlights of this section:
> 1. Introduce the compilation design of the kernel.
> 2. Introduce the establishment of kernel dependencies.

##### 2.3.5.1 Kernel compilation

After the original OpKernel is migrated to PHI, phi automatically scans all relevant cc(cu) files during compilation, and compiles the overall target according to the device, without declaring the Kernel compilation objects one by one, for example:

```
file(
  GLOB
  kernel_cu
  "gpu/*.cu"
  "gpu/*.cu.cc"
  "gpudnn/*.cu"
  "kps/*.cu"
  "selected_rows/gpu/*.cu"
  "sparse/gpu/*.cu"
  "strings/*.cu"
  "strings/gpu/*.cu")

add_library(phi_cpu ${kernel_cc})
kernel_declare("${kernel_cc}")
target_link_libraries(phi_cpu ${COMMON_KERNEL_DEPS})
```

By calling the `kernel_declare` method, the registration unit in the kernel source file is extracted, and a unified symbol declaration is automatically generated to avoid manual maintenance of the kernel declaration. The generated declaration is in the `paddle/phi/kernels/declarations.h` file in the `build` directory, the generated declaration code example is as follows:

```c++
PD_DECLARE_KERNEL(argsort, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(as_complex, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(as_real, CPU, ALL_LAYOUT);
...
```

For the specific implementation of `kernel_declare`, please refer to the function implementation in `camke/phi.cmake`, which will not be introduced here.

##### 2.3.5.2 Kernel dependencies

The phi kernel has been changed to a functional format, and the original intention is to make it easier to reuse between kernels, but reusing kernels will introduce compilation dependencies between kernels. We compile all kernels as a whole unit, which can avoid maintaining dependencies between individual kernels. Therefore, if you need to reuse the Kernel during development, you only need to include the corresponding header file correctly.

#### 2.3.6 InferMeta Abstract Integration

The original `InferShape` of fluid Op is the same as `OpKernel`, has the problem of repeated development: because the `InferShape` functions of different Ops cannot be reused, even if the `InferShape` logic of different Ops is the same or similar, they needs to be rewritten again. The refactor of PHI needs to address this issue.

We also rewrite `InferShape` into a functional form, which supports different Ops to call the same `InferShape` function, which improves ease of use and reduces maintenance costs.

> FAQ:
> 1. Why call it `InferMeta` instead of continuing to call it `InferShape`?
> Answer: The `Meta` of `InferMeta` comes from the `meta` member in `DenseTensor`. In PHI, an op has two components, `InferMeta` and `Kernel`. `InferMeta` covers the functions of `InferShape`, but it is not limited to `InferShape`. In addition to the inference of dims and lod, `InferMeta` also infers dtype and layout, which is different from the original.

##### 2.3.6.1 InferMeta related design

`InferMeta` is also functional form, a few examples are as follows:

```c++
void UnchangedInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->share_meta(x);
}

void CastInferMeta(const MetaTensor& x, DataType out_dtype, MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(out_dtype);
  out->set_layout(x.layout());
}

void CreateLikeInferMeta(const MetaTensor& x,
                         DataType dtype,
                         DataLayout layout,
                         MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(dtype == DataType::UNDEFINED ? x.dtype() : dtype);
  out->set_layout(layout == DataLayout::UNDEFINED ? x.layout() : layout);
}

void ConcatInferMeta(const std::vector<MetaTensor>& x,
                     const Scalar& axis_scalar,
                     MetaTensor* out,
                     MetaConfig config = MetaConfig());
```

The features are introduced as follows:

1. The function is named `[FunctionDesc|OpName]InferMeta`
2. The function form is similar to Kernel, the function parameters are MetaTensor input, Attribute, MetaTensor output in turn, and the return value is empty. In principle, there is a one-to-one correspondence between the parameter list of the `InferMeta` function and its corresponding `Kernel` function. The difference is only the Tensor parameter type. The Tensor parameter of the `InferMeta` function is `MetaTensor`, and the Tensor parameter of the `Kernel` function is `DenseTensor`, `SparseTensor`, etc.
3. For some `InferMeta` functions that need to distinguish between compile time and execution time, add the `MetaConfig` parameter at the end. There is a bool member `is_runtime` in config, and the structure is used to facilitate the subsequent expansion of other flag members.

The purpose of using `MetaTensor` is to mask multiple Tensor types, and to be compatible with the original fluid's `VarDesc` and `Variable`. One op corresponds to one `InferMeta` function. If the type is not masked, the `InferMeta` function will be overloaded multiple times for different input types.

The basic design of `MetaTensor` see the `paddle/phi/core/meta_tensor.h`. There is a pointer member `TensorBase` in the base class `MetaTensor`, so it can be compatible with `DenseTensor`, `SelectedRows`, `SparseCsrTensor` and other types in PHI.


> Note:
> Only the content related to the design of PHI itself in this README. If you want to know more about the design of how phi and fluid are compatible, please refer to:
> 1. [Paddle HIgh reusability operator library (PHI) Design Document (CN Version)](https://github.com/PaddlePaddle/docs/blob/develop/docs/design/phi/design_cn.md)
> 2. [Paddle HIgh reusability operator library (PHI) Design Document (EN Version)](https://github.com/PaddlePaddle/docs/blob/develop/docs/design/phi/design_en.md)
