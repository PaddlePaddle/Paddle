# 如何支持一个新的设备

## 概览

添加一个新的设备需要以下3个步骤：

* [在`CMakeList`中添加设备的支持](#0001)
* [在`saber`中添加设备的实现](#0002)
* [在`framework`中添加设备的具体化或实例化](#0003)

假设新设备的名称为`TNEW`, 以下将以这个设备名称进行演示。

## <span id = '0001'> 在`CMakeList`中添加设备的支持 </span> ##

* 修改根目录`CMakeList.txt`
```cmake
#select the plantform to build
anakin_option(USE_GPU_PLACE "Select the build mode for GPU place." NO)
anakin_option(USE_X86_PLACE "Select the build mode for X86 place." NO)
anakin_option(USE_ARM_PLACE "Select the build mode for ARM place." NO)
anakin_option(USE_TNEW_PLACE "Select the build mode for ARM place." YES)
```

* 修改`saber/CMakeList.txt`

根据新增设备的目录完善`saber`目录下的`CMakeList.txt`。
```cmake
if(USE_TNEW_PLACE)
    anakin_fetch_files_with_suffix(${ANAKIN_SABER}/core/impl/tnew "cpp" ANAKIN_SABER_BASE_SRC)
    anakin_fetch_files_with_suffix(${ANAKIN_SABER}/funcs/impl/tnew "cpp" ANAKIN_SABER_BASE_SRC)
endif()
```

* 修改`test/CMakeList.txt`

新增设备的单测文件放在`test/saber/tnew`目录下，修改`test`目录下的`CMakeList.txt`。
```cmake
if(USE_TNEW_PLACE)
    anakin_fetch_files_with_suffix(${ANAKIN_UNIT_TEST}/saber/tnew "cpp" ANAKIN_TEST_CASE_SRC)
endif()
```

* 修改`cmake/anakin_config.h.in`
```c++
// plantform to use
#cmakedefine USE_GPU_PLACE

#cmakedefine USE_X86_PLACE

#cmakedefine USE_ARM_PLACE

#cmakedefine USE_TNEW_PLACE
```

* 其他依赖和编译选项    
修改`cmake`目录下的`compiler_options.cmake`和`find_modules.cmake`


## <span id = '0002'> 在`saber`中添加设备的实现 </span> ##
`saber`是`Anakin`的基础计算库，对外提供设备无关的统一的API，设备相关的实现都会封装到`TargetWrapper`中。

### 在`saber/saber_types.h`中添加设备

```c++
enum TargetTypeEnum {
    eINVALID = -1,
    eNV = 1,
    eAMD = 2,
    eARM = 3,
    eX86 = 4,
    eNVHX86 = 5,
    eTNEW = 6
};

typedef TargetType<eNV> NV;
typedef TargetType<eARM> ARM;
typedef TargetType<eAMD> AMD;
typedef TargetType<eX86> X86;
typedef TargetType<eTNEW> TNEW;

```

### 在`saber/core`中添加设备的实现

1. 在`target_traits.h`中添加新设备

* 增加设备类型
```c++
struct __cuda_device{};
struct __arm_device{};
struct __amd_device{};
struct __x86_device{};
struct __tnew_device{};
```

* `TargetTypeTraits`模板具体化
```c++
template <>
struct TargetTypeTraits<TNEW> {
    typedef __xxx_target target_category;//根据实际设备是host端还是device端进行选择
    typedef __tnew_device target_type;
};
```

2. 在`data_traits.h`中特化`DataTrait`模板类

如果设备需要特殊的数据类型，则特化出设备的`DataTrait`类的实现，例如opencl数据类型的实现如下：
```c++
#ifdef USE_OPENCL
struct ClMem{
    ClMem(){
        dmem = nullptr;
        offset = 0;
    }

    ClMem(cl_mem* mem_in, int offset_in = 0) {
        dmem = mem_in;
        offset = offset_in;
    }

    ClMem(ClMem& right) {
        dmem = right.dmem;
        offset = right.offset;
    }

    ClMem& operator=(ClMem& right) {
        this->dmem = right.dmem;
        this->offset = right.offset;
        return *this;
    }

    ClMem& operator+(int offset_in) {
        this->offset += offset_in;
        return *this;
    }

    int offset{0};
    cl_mem* dmem;
};

template <>
struct DataTrait<AMD, AK_FLOAT> {
    typedef ClMem Dtype;
    typedef float dtype;
};

template <>
struct DataTrait<AMD, AK_DOUBLE> {
    typedef ClMem Dtype;
    typedef double dtype;
};

template <>
struct DataTrait<AMD, AK_INT8> {
    typedef ClMem Dtype;
    typedef char dtype;
};
#endif //use_opencl
```

3. 在`target_wrapper.h`中特化`TargetWrapper`模板类

特化`TargetWrapper`模板类，在`target_wrapper.h`中声明函数，具体如下：
```c++
template <>
struct TargetWrapper<TNEW, __xxx_target> { //根据TNEW的具体类型修改__xxx_target，__host_target或者__device_target

    typedef xxx_event event_t;          //根据设备实现xxx_event
    typedef xxx_stream stream_t;        //根据设备实现xxx_stream

    static void get_device_count(int& count);

    static void set_device(int id);

    //We should add strategy to avoid malloc directly
    static void mem_alloc(void** ptr, size_t n);

    static void mem_free(void* ptr);

    static void mem_set(void* ptr, int value, size_t n);

    static void create_event(event_t& event, bool flag = false);

    static void create_stream(stream_t& stream);

    static void create_stream_with_flag(stream_t& stream, unsigned int flag);

    static void create_stream_with_priority(stream_t& stream, unsigned int flag, int priority);

    static void destroy_stream(stream_t& stream);

    static void destroy_event(event_t& event);

    static void record_event(event_t& event, stream_t stream);

    static void query_event(event_t& event);

    static void sync_event(event_t& event);

    static void sync_stream(event_t& event, stream_t& stream);

    static void sync_memcpy(void* dst, int dst_id, const void* src, int src_id, \
                            size_t count, __DtoD);

    static void async_memcpy(void* dst, int dst_id, const void* src, int src_id, \
                             size_t count, stream_t& stream, __DtoD);

    static void sync_memcpy(void* dst, int dst_id, const void* src, int src_id, \
                            size_t count, __HtoD);

    static void async_memcpy(void* dst, int dst_id, const void* src, int src_id, \
                             size_t count, stream_t& stream, __HtoD);

    static void sync_memcpy(void* dst, int dst_id, const void* src, int src_id, \
                            size_t count, __DtoH);

    static void async_memcpy(void* dst, int dst_id, const void* src, int src_id, \
                             size_t count, stream_t& stream, __DtoH);

    static void sync_memcpy_p2p(void* dst, int dst_dev, const void* src, \
                                int src_dev, size_t count);

    static void async_memcpy_p2p(void* dst, int dst_dev, const void* src, \
                                 int src_dev, size_t count, stream_t& stream);

    static int get_device_id();
};

```

4. 在`impl/`目录下添加设备目录和实现

在`saber/core/impl`目录下添加设备目录`tnew`。
* 实现`TargetWrapper<TNEW, __xxx_target>`结构体中各函数的定义。    
如果`TargetWrapper<TNEW, __xxx_target>`的实现与默认的模板类一致，则不用特化出该类。

```c++
typedef TargetWrapper<TNEW, __xxx_target> TNEW_API;
void TNEW_API::get_device_count(int &count) {
    // add implementation
}

void TNEW_API::set_device(int id){
    // add implementation
}
        
void TNEW_API::mem_alloc(void** ptr, size_t n){
    // add implementation
}
        
void TNEW_API::mem_free(void* ptr){
    if(ptr != nullptr){
        // add implementation
    }
}
...

```

* 特化实现`device.h`中的`Device<TNEW>`

```c++
template <>
void Device<TNEW>::create_stream() {
    // add implementation
}

template <>
void Device<TNEW>::get_info() {

    // add implementation
}

```

### 在`saber/funcs`中实现设备相关的op

参考[如何增加新的Operator](addCustomOp.md)


## <span id = '0003'> 在`framework`中添加设备的具体化或实例化 </span> ##

### `framework/core`

* `net.cpp`中添加实例化

```c++
#ifdef USE_TNEW_PLACE
template class Net<TNEW, AK_FLOAT, Precision::FP32, OpRunType::ASYNC>;
template class Net<TNEW, AK_FLOAT, Precision::FP32, OpRunType::SYNC>;
#endif
```

* `operator_func.cpp`中添加实例化

```c++
#ifdef USE_TNEW_PLACE
template class OperatorFunc<TNEW, AK_FLOAT, Precision::FP32>;
#endif
```

* `worker.cpp`中添加实例化

```c++
#ifdef USE_TNEW_PLACE
template class Worker<TNEW, AK_FLOAT, Precision::FP32, OpRunType::ASYNC>;
template class Worker<TNEW, AK_FLOAT, Precision::FP32, OpRunType::SYNC>;
#endif
```

* `operator_attr.cpp`中添加实例化

```c++
template
OpAttrWarpper& OpAttrWarpper::__alias__<TNEW, AK_FLOAT, Precision::FP32>(const std::string& op_name);
template
OpAttrWarpper& OpAttrWarpper::__alias__<TNEW, AK_FLOAT, Precision::FP16>(const std::string& op_name);
template
OpAttrWarpper& OpAttrWarpper::__alias__<TNEW, AK_FLOAT, Precision::INT8>(const std::string& op_name);
```

* `parameter.h`中添加设备的实现

```c++
#ifdef USE_TNEW_PLACE
template<typename Dtype>
class PBlock<Dtype, TNEW> {
public:
	typedef Tensor4d<TNEW, DataTypeRecover<Dtype>::type> type;

	PBlock() {
		_inner_tensor = std::make_shared<type>(); 
	}
	...
}
#endif //TNEW
```

* `type_traits_extend.h`中添加设备的实现

```c++
template<>
struct target_host<saber::TNEW> {
    typedef saber::X86 type; //根据TNEW选择正确的host type
};
```

### `framework/graph`

* `graph.cpp`中添加实例化
  
```c++
  #ifdef USE_TNEW_PLACE
  template class Graph<TNEW, AK_FLOAT, Precision::FP32>;
  template class Graph<TNEW, AK_FLOAT, Precision::FP16>;
  template class Graph<TNEW, AK_FLOAT, Precision::INT8>;
  #endif
```

### `framework/model_parser`

* `parser.cpp`中添加实例化
  
```c++
  #ifdef USE_TNEW_PLACE
  template
  Status load<TNEW, AK_FLOAT, Precision::FP32>(graph::Graph<TNEW, AK_FLOAT, Precision::FP32>* graph,
          const char* model_path);
  template
  Status load<TNEW, AK_FLOAT, Precision::FP16>(graph::Graph<TNEW, AK_FLOAT, Precision::FP16>* graph,
          const char* model_path);
  template
  Status load<TNEW, AK_FLOAT, Precision::INT8>(graph::Graph<TNEW, AK_FLOAT, Precision::INT8>* graph,
          const char* model_path);
  
  template
  Status save<TNEW, AK_FLOAT, Precision::FP32>(graph::Graph<TNEW, AK_FLOAT, Precision::FP32>* graph,
          std::string& model_path);
  template
  Status save<TNEW, AK_FLOAT, Precision::FP16>(graph::Graph<TNEW, AK_FLOAT, Precision::FP16>* graph,
          std::string& model_path);
  template
  Status save<TNEW, AK_FLOAT, Precision::INT8>(graph::Graph<TNEW, AK_FLOAT, Precision::INT8>* graph,
          std::string& model_path);
  
  template
  Status load<TNEW, AK_FLOAT, Precision::FP32>(graph::Graph<TNEW, AK_FLOAT, Precision::FP32>* graph,
          std::string& model_path);
  template
  Status load<TNEW, AK_FLOAT, Precision::FP16>(graph::Graph<TNEW, AK_FLOAT, Precision::FP16>* graph,
          std::string& model_path);
  template
  Status load<TNEW, AK_FLOAT, Precision::INT8>(graph::Graph<TNEW, AK_FLOAT, Precision::INT8>* graph,
          std::string& model_path);
  
  template
  Status save<TNEW, AK_FLOAT, Precision::FP32>(graph::Graph<TNEW, AK_FLOAT, Precision::FP32>* graph,
          const char* model_path);
  template
  Status save<TNEW, AK_FLOAT, Precision::FP16>(graph::Graph<TNEW, AK_FLOAT, Precision::FP16>* graph,
          const char* model_path);
  template
  Status save<TNEW, AK_FLOAT, Precision::INT8>(graph::Graph<TNEW, AK_FLOAT, Precision::INT8>* graph,
          const char* model_path);
  #endif
```

* `model_io.cpp`中添加实例化

```c++
#ifdef USE_TNEW_PLACE
template class NodeIO<TNEW, AK_FLOAT, Precision::FP32>;
template class NodeIO<TNEW, AK_FLOAT, Precision::FP16>;
template class NodeIO<TNEW, AK_FLOAT, Precision::INT8>;
#endif
```

### `framework/operators`

为`framework/operators`目录下所有op添加实例化或具体化
以`activation.cpp`为例，实例化如下：

```c++
#ifdef USE_TNEW_PLACE
INSTANCE_ACTIVATION(TNEW, AK_FLOAT, Precision::FP32);
INSTANCE_ACTIVATION(TNEW, AK_FLOAT, Precision::FP16);
INSTANCE_ACTIVATION(TNEW, AK_FLOAT, Precision::INT8);
template class ActivationHelper<TNEW, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Activation, ActivationHelper, TNEW, AK_FLOAT, Precision::FP32);
#endif
```

如果TNEW设备函数的实现与现有模板实现不一致，可以特化实现如下（以init()为例）：
```c++
#ifdef USE_TNEW_PLACE
INSTANCE_ACTIVATION(TNEW, AK_FLOAT, Precision::FP32);
INSTANCE_ACTIVATION(TNEW, AK_FLOAT, Precision::FP16);
INSTANCE_ACTIVATION(TNEW, AK_FLOAT, Precision::INT8);
template <>
Status ActivationHelper<TNEW, AK_FLOAT, Precision::FP32>::Init(OpContext<TNEW> &ctx,\
        const std::vector<Tensor4dPtr<TNEW, AK_FLOAT> >& ins, \
                std::vector<Tensor4dPtr<TNEW, AK_FLOAT> >& outs) {
    SABER_CHECK(_funcs_activation.init(ins, outs, _param_activation, SPECIFY, SABER_IMPL, ctx)); //在这里选择实现方式
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(Activation, ActivationHelper, TNEW, AK_FLOAT, Precision::FP32);
#endif
```

在`ANAKIN_REGISTER_OP(Activation)`中添加TNEW的注册

```c++
#ifdef USE_TNEW_PLACE
.__alias__<TNEW, AK_FLOAT, Precision::FP32>("activation")
#endif
```

## 注意事项
不要修改`Tensor`/`Buffer`/`Env`/`Context`这些类函数的接口和实现
