# 如何增加新的Operator

## 基本概念

简单介绍下几个同Operator相关的基本概念，详情请参考设计文档。

```framework```: 上层的逻辑代码，负责从parser中获取参数及weights，添加op时主要修改framework/operator目录下的内容。

```saber```: 底层的实现代码，Anakin通过saber封装了不同的backends，不同的实现(impl)分别特化出自己的实现，外层framework通过不同的template进入各自的impl完成调用。各个op的parameter放在saber/saber_funcs_param.h文件中，增加op主要修改saber/funcs下的内容。

saber的文件结构：
* saber/funcs下的是各个funcs的外部接口，这一层的op与具体的设备实现无关，只与各op完成的功能有关。由于跟实现(impl)无关，本层文件明均不带impl。
* saber/funcs/impl下是各个op的impl声明，特定设备需要完成该层声明的特化版本，如saber/funcs/impl/x86实现了上一层impl声明的x86特化版本，saber/funcs/impl/cuda实现了上一层impl声明的NV特化版本。当增加新的backends时需要特化出新的实现。本层代码同实现相关，均带有```impl_```前缀。
* saber/funcs/impl/cuda/base/cuda_c内有cuda```.cu```扩展名的文件，添加cuda的kernel需要在该文件目录下添加。
* saber/funcs/impl/cuda/base/sass 内有不同架构的汇编代码编译的静态库。

### 涉及到的基类及各个类之前的关系

简单介绍相关的基类

* ```anakin::Operator```: framework的operator基类，位于framework/core/operator/operator.h

* ```anakin::saber::BaseFunc```: saber对外的op接口基类，提供统一的对外接口，位于saber/funcs/base.h。BaseFunc的```compute_output_shape```接口只根据input的shape和param的参数计算输出的shape，并通过```tensor```的```set_shape```接口(只设置shape，不分配空间)设置到output中。```operator()```接口为各个op的计算接口。

* ```ankain::saber::ImplBase```: saber设备实现的op的接口，所有设备相关实现的基类。位于saber/funcs/impl/impl_base.h。实现版本中这里分为两类，一类以```vender_```为前缀，带有```vender_```代码意为使用第三方库来实现该op，如cudnn的conv，或mkl的conv等等，这类op的性能我们难以调优，因此单独列为一类。另一类是带有源码的saber实现，这些实现都带有```saber_```为前缀，此类实现带有源码，能够通过后续优化不断提升性能，实现起名时需要注意这一点。

## 添加operator

添加一个新的op需要以下几步：

1. 添加saber的param
2. 定义saber的Operator类
3. 定义新的impl声明
3. 完成新的impl实现
4. 增加framework的实现或特化

接下来就针对这几步，以一个简单例子为例介绍实现。

例如我们要添加新的Mul op。给出计算公式如下：$$Out = alpha \dot X * Y$$

### 为operator增加param

涉及到的文件：```saber/saber_funcs_param.h```。如果之前已经存在需要添加的op的param，这一步可以跳过。
这里```XXXParam```是一个```struct```。包含一个无参数的构造函数，含参数的构造函数，复制构造函数，```operator=()```及```operator==()```。
```
template <typename opTensor> // 能够获得target, datatype, layout
struct MulParam{
  MulParam()
    : alpha(0)
  {}
  MulParam(float alpha_in)
    : alpha(alpha_in)
  {}
  MulParam(const MulParam& right)
    : alpha(right.alpha)
  {}
  MulParam &operator=(const MulParam &right) {
    alpha = right.alpha;
  }
  bool operator==(const MulParam &right) {
    return alpha == right.alpha;
  }
  float alpha;
};
```

### 定义Operator类
涉及到的文件:```saber/funcs/mul.h```。如果之前定义过该op的类，这里需要修改输入的impl定义头文件。
下面给出一个相对完整的定义结构供参考。
```
//不同的设备需要包含对应的operator实现.[详见](#impl)
#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_mul.h"
#include "saber/funcs/impl/cuda/vender_mul.h"
#endif
//如果一个设备现在还没有对应的operator实现，需要包含声明。[详见](#declare)
#ifdef USE_X86_PLACE
#include "saber/funcs/impl/impl_mul.h"
#endif
namespace anakin {
namespace saber {
template<typename TargetType,
        DataType OpDtype,
        DataType inDtype = AK_FLOAT,
        DataType outDtype = AK_FLOAT,
        typename LayOutType_op = NCHW,
        typename LayOutType_in = NCHW,
        typename LayOutType_out = NCHW>
class Mul : public BaseFunc<
        Tensor<TargetType, inDtype, LayOutType_in>,
        Tensor<TargetType, outDtype, LayOutType_out>,
        Tensor<TargetType, OpDtype, LayOutType_op>,
        ImplBase, MulParam> {
public:
    using BaseFunc<
            Tensor<TargetType, inDtype, LayOutType_in>,
            Tensor<TargetType, outDtype, LayOutType_out>,
            Tensor<TargetType, OpDtype, LayOutType_op>,
            ImplBase, MulParam>::BaseFunc;
    Mul() = default;
    typedef Tensor<TargetType, inDtype, LayOutType_in> InDataTensor;
    typedef Tensor<TargetType, outDtype, LayOutType_out> OutDataTensor;
    typedef Tensor<TargetType, OpDtype, LayOutType_op> OpTensor;
    typedef MulParam<OpTensor> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v &input,
                                             Output_v &output, Param_t &param) override {
        //计算输出的shape，
        Shape output_shape = (input[0]->valid_shape());
        /* code */
        return output[0]->set_shape(output_shape);
    }
    virtual SaberStatus init_impl(ImplEnum implenum) override {
      // 不同设备均使用此init_impl, 此接口创建对应impl的实现。
      switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderMul <TargetType,
                OpDtype, inDtype, outDtype,
                LayOutType_op, LayOutType_in, LayOutType_out>);
                return SaberSuccess;
            case SABER_IMPL:
                this->_impl.push_back(new SaberMul <TargetType,
                OpDtype, inDtype, outDtype,
                LayOutType_op, LayOutType_in, LayOutType_out>);
                return SaberSuccess;
            default:
                return SaberUnImplError;
        }
    }
private:
    virtual void pick_best_static() override {
        if (true) // some condition?
            this->_best_impl = this->_impl[0];
    }
    virtual void pick_best_specify(ImplEnum implenum) override {
        this->_best_impl = this->_impl[0];
    }
};
} // namespace saber
} // namespace anakin
```

### 为operator增加新的impl<span id="declare">声明</span>

涉及的文件:```saber/funcs/impl/impl_mul.h```。不同的设备都特化同一个声明，特化版本放在对应的文件夹下，这里的声明就是给出所有设备的统一声明。下面给出一个参考。
```
#include "saber/funcs/impl/impl_macro.h"
namespace anakin{
namespace saber{
DEFINE_OP_CLASS(Mul, MulParam); // 第一个参数是op的名字，第二个是对应param的名字
}
}
```

### 完成新的operator特定后端<span id="impl">实现</span>

涉及的文件:```saber/funcs/impl/xxx/vender_mul.h```或```saber/funcs/impl/xxx/saber_mul.h```
这里```xxx```指代特定的一种设备。```vender```是指的使用第三方库实现的op，```saber```指的源码实现的op。这里以cuda的vender实现为例，简单介绍一下特化出的函数的几个基本接口。

```
// include 对应的声明
#include "saber/funcs/impl/impl_mul.h"

namespace anakin{
namespace saber{
template <DataType OpDtype,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
class VenderMul<NV, //偏特化出需要的后端。
    OpDtype, inDtype, outDtype,
    LayOutType_op, LayOutType_in, LayOutType_out> :
    public ImplBase<
        Tensor<NV, inDtype, LayOutType_in>,
        Tensor<NV, outDtype, LayOutType_out>,
        Tensor<NV, OpDtype, LayOutType_op>,
        MulParam<Tensor<NV, OpDtype, LayOutType_op> > >
{
public:
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<NV, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;
    VenderMul(){}
    ~VenderMul() {}

    virtual SaberStatus init(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            MulParam<OpTensor>& param, Context<NV>& ctx) {
        this->_ctx = ctx;
        create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            MulParam<OpTensor>& param, Context<NV>& ctx) {
        // set内部参数
    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                          std::vector<DataTensor_out*>& outputs,
                        MulParam<OpTensor>& param) {
        // dispatch kernel.
    }

private:
};
}
}
```
```init```和```create```的区别：```init```接口是第一次初始化op的时候进入的接口，此函数只在第一次初始化op时调用，这个接口一般放一些只需要执行一次的代码，如malloc或者create之类的函数。```create```函数除了第一次init执行外，在输入发生变化或者param发生变化时会再次触发，create一般放置set函数，设置内部变量，当input发生变化时这里执行一些同input或weights直接相关的代码。但create因为触发位置在网络内，如果```create```函数执行了一些严重耗时的操作，这里会拖慢整个op的执行时间，需要慎重选择操作放置的位置。
### 添加framework的特化

涉及的文件:```framework/operators/mul.h```和```framework/operators/mul.cpp```。
这里简单介绍下如果添加或修改framework内的operator

```
#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/mul.h" // 需要包对应的saber头文件
namespace anakin {
namespace ops {
template<typename Ttype, DataType Dtype, Precision Ptype>
class MulHelper;

template<typename Ttype, DataType Dtype, Precision Ptype>
class Mul : public Operator<Ttype, Dtype, Ptype> {
public:
    Mul() {}
    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx,
                             const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
                             std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
        LOG(ERROR) << "Not Impl Yet Operator power<TargetType:"<<"unknown"<<","
                   <<type_id<typename DataTypeWarpper<Dtype>::type>().type_info()<<">";
    }
    friend class MulHelper<Ttype, Dtype, Ptype>;
};
template<typename Ttype, DataType Dtype, Precision Ptype>
class MulHelper : public OperatorHelper<Ttype, Dtype, Ptype> {
public:
    MulHelper() = default;
    ~MulHelper();
    Status InitParam() override;

    Status Init(OpContext<Ttype> &ctx,
                const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
                std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override;
    Status InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
                      std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override;

public:
    saber::MulParam<Tensor4d<Ttype, Dtype>> _param_mul;
    saber::Mul<Ttype, Dtype> _funcs_mul;
};
}
} /* namespace anakin */
```
对应的```.cpp```文件如下：
```
#include "framework/operators/mul.h"

namespace anakin {
namespace ops {

#ifdef USE_CUDA
template<>
void Mul<NV, AK_FLOAT, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins,
    std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
    auto* impl =
        static_cast<MulHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param =
        static_cast<MulHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper)->_param_mul;
    impl->_funcs_mul(ins, outs, param, ctx);
}
#endif

template<typename Ttype, DataType Dtype, Precision Ptype>
Status MulHelper<Ttype, Dtype, Ptype>::InitParam() {
    auto alpha = GET_PARAMETER(float, alpha);
    MulParam<Tensor4d<Ttype, Dtype>> param_mul(alpha);
    _param_mul = param_mul;
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status MulHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {

    SABER_CHECK(_funcs_mul.init(ins, outs, _param_mul, SPECIFY, VENDER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status MulHelper<Ttype, Dtype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_mul.compute_output_shape(ins, outs, _param_mul));
    return Status::OK();
}

#ifdef USE_CUDA
template class MulHelper<NV, AK_FLOAT, Precision::FP32>;
#endif
#ifdef USE_ARM_PLACE
template class MulHelper<ARM, AK_FLOAT, Precision::FP32>;
#endif
// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(Mul, MulHelper, NV, AK_FLOAT, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(Mul, MulHelper, ARM, AK_FLOAT, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(Mul)
.Doc("Mul operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("mul")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("mul")
#endif
.num_in(1)
.num_out(1)
.Args<float>("alpha", " alpha of Mul "); //注册

} /* namespace ops */

} /* namespace anakin */
```

## 实现单元测试
涉及的文件:```test/saber/xxx/test_saber_funcs_mul_xxx.cpp```
在对应的test下需要添加新的单元测试

```
TEST(TestSaberFuncNV, test_depthwise_conv) {

    // init tensors and some param.

    // start Reshape & doInfer
    Context<NV> ctx1(0, 1, 1);

    // create param
    MulParam<Tensor<NV, AK_FLOAT, NCHW> > param(alpha);

    std::vector<Tensor<NV, AK_FLOAT, NCHW>*> input;
    std::vector<Tensor<NV, AK_FLOAT, NCHW>*> output;

    // create saber op
    Mul<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> mul;

    // compute output shape
    mul.compute_output_shape(input, output, param);

    // re_alloc output tensors memory based on output shape
    output[0]->re_alloc(output[0]->shape());

    // init saber op(calling init and create)
    mul.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);

    // call operator()
    mul(input, output, param, ctx1);

    // cuda specified, record events
    cudaStream_t cuda_stream = ctx1.get_compute_stream();
    output[0]->record_event(cuda_stream);
    output_dev.sync();
    
    // param changed 
    param.alpha = 2.0;
    // auto calling saber op(create and dispatch)
    mul(input, output, param, ctx1);

    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());
}

int main(int argc, const char** argv){
    anakin::saber::Env<NV>::env_init();

    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

```
## 调试及注意事项

一个op需要有对外的op接口和内部实现，由于存在saber/funcs/impl的非特化版本声明，当有op在某种设备下没有对应实现时，也能够编译，但此时是没有任何实现的空实现，
