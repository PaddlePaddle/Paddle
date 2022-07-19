#include "generic_plugin.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/compat/op_utils.h"
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/fluid/framework/framework.pb.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

platform::CUDAPlace GenericPlugin::place_;
// paddle::platform::CUDADeviceContext GenericPlugin::dev_ctx_(GenericPlugin::place_);
paddle::memory::allocation::CUDAAllocator GenericPlugin::allocator_(place_);

phi::Attribute Convert2PhiAttribute(framework::OpDesc& op_desc, std::string attr_name) {
  auto attr_type = op_desc.GetAttrType(attr_name);
  switch (attr_type) {
    case framework::proto::AttrType::INT: {
      return phi::Attribute(PADDLE_GET_CONST(int, op_desc.GetAttr(attr_name)));
    };
    case framework::proto::AttrType::FLOAT: {
      return phi::Attribute(PADDLE_GET_CONST(float, op_desc.GetAttr(attr_name)));
    };
    case framework::proto::AttrType::STRING: {
      return phi::Attribute(PADDLE_GET_CONST(std::string, op_desc.GetAttr(attr_name)));
    };
    case framework::proto::AttrType::INTS: {
      return phi::Attribute(PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr(attr_name)));
    };
    case framework::proto::AttrType::FLOATS: {
      return phi::Attribute(PADDLE_GET_CONST(std::vector<float>, op_desc.GetAttr(attr_name)));
    };
    case framework::proto::AttrType::STRINGS: {
      return phi::Attribute(PADDLE_GET_CONST(std::vector<std::string>, op_desc.GetAttr(attr_name)));
    };
    case framework::proto::AttrType::BOOLEAN: {
      return phi::Attribute(PADDLE_GET_CONST(bool, op_desc.GetAttr(attr_name)));
    };
    case framework::proto::AttrType::BOOLEANS: {
      return phi::Attribute(PADDLE_GET_CONST(std::vector<bool>, op_desc.GetAttr(attr_name)));
    };
    default:{
      LOG(ERROR) << "Can't conver op's attribute ["<< attr_name << "] to phi Attribute.";
    }
  }
}

void GenericPlugin::InferShape(const paddle::framework::proto::OpDesc &proto_op_desc, const nvinfer1::Dims* inputDims, int nbInputDims) {
  assert(nbInputDims == op_desc_.OutputNames().size());

  auto dims_2_vector = [](const nvinfer1::Dims &dims) -> std::vector<int> {
                            std::vector<int>res;
                            if(dims.nbDims == 3)
                              res.push_back(1); // batch_size
                            for(int i = 0; i < dims.nbDims; i++) {
                              res.push_back(dims.d[i]);
                            }
                            return res;
                        };
  
  auto ddim_2_Nvdims = [](const phi::DDim ddim) -> nvinfer1::Dims {
                            nvinfer1::Dims res;
                            res.nbDims = ddim.size();
                            for(int i = 0; i < ddim.size(); i++) {
                              res.d[i] = ddim[i];
                            }
                            return res;
                        };

  phi::InferMetaContext ctx;
  std::vector<phi::DenseTensor>input_dense_tensors;
  std::vector<phi::MetaTensor>input_meta_tensors;
  for(int i = 0; i < nbInputDims; i++) {
    input_dense_tensors.emplace_back();
    input_dense_tensors.back().Resize(phi::make_ddim(dims_2_vector(inputDims[i])));
    input_meta_tensors.emplace_back(&input_dense_tensors.back());
    ctx.EmplaceBackInput(input_meta_tensors.back());
  }
  
  std::vector<phi::DenseTensor>output_dense_tensors;
  std::vector<phi::MetaTensor>output_meta_tensors;
  for(int i = 0; i < getNbOutputs(); i++) {
    output_dense_tensors.emplace_back();
    output_meta_tensors.emplace_back(&output_dense_tensors.back());
    ctx.EmplaceBackOutput(output_meta_tensors.back());
  }
  
  std::string op_type = op_desc_.Type();
  LOG(INFO) << "op_type : " << op_type;
  PADDLE_ENFORCE_EQ(
        phi::OpUtilsMap::Instance().HasArgumentMappingFn(op_type),
        true,
        platform::errors::Fatal(
            "%s has no argument mapping function!.", op_type.c_str()));
  const phi::ArgumentMappingFn* argument_mapping_func = phi::OpUtilsMap::Instance().GetArgumentMappingFn(op_type);
  PADDLE_ENFORCE_NOT_NULL(
        argument_mapping_func,
        platform::errors::Fatal(
            "ArgumentMappingFn should not be null!."));
  PluginArgumentMappingContext argument_mapping_context(&op_desc_);
  phi::KernelSignature pt_kernel_signature = (*argument_mapping_func)(argument_mapping_context);
  for(int i = 0; i < pt_kernel_signature.attr_names.size(); i++) {
    ctx.EmplaceBackAttr(Convert2PhiAttribute(op_desc_, pt_kernel_signature.attr_names[i]));
    LOG(INFO) << pt_kernel_signature.attr_names[i];
  }
  ctx.SetMetaConfig({/*is_runtime =*/true, /*is_run_mkldnn_kernel=*/false});
  phi::MetaFnFactory::Instance().Get(op_desc_.Type())(&ctx);
  
  for(int i = 0; i < getNbOutputs(); i++) {
    output_dims_.push_back(ddim_2_Nvdims(output_dense_tensors[i].dims()));
  }
  for(int i = 0; i < output_dims_[0].nbDims; i++) {
    LOG(INFO) << output_dims_[0].d[i];
  }
}

int GenericPlugin::initialize() TRT_NOEXCEPT { 
  proto_op_desc_.ParseFromString(op_meta_data_);
  op_desc_ = std::move(framework::OpDesc(proto_op_desc_, nullptr));
  std::string op_type = op_desc_.Type();

  PADDLE_ENFORCE_EQ(
        phi::OpUtilsMap::Instance().HasArgumentMappingFn(op_type),
        true,
        platform::errors::Fatal(
            "%s has no argument mapping function!.", op_type.c_str()));

  const phi::ArgumentMappingFn* argument_mapping_func = phi::OpUtilsMap::Instance().GetArgumentMappingFn(op_type);
  PluginArgumentMappingContext argument_mapping_context(&op_desc_);
  phi::KernelSignature pt_kernel_signature = (*argument_mapping_func)(argument_mapping_context);

  paddle::framework::OpKernelType kernel_type(paddle::framework::proto::VarType_Type_FP32, place_);
  phi::KernelKey pt_kernel_key = paddle::framework::TransOpKernelTypeToPhiKernelKey(kernel_type);

  PADDLE_ENFORCE_EQ(
        phi::KernelFactory::Instance().HasCompatiblePhiKernel(op_type),
        true,
        platform::errors::Fatal(
            "%s has no compatible phi kernel!.", op_type.c_str()));
  const phi::Kernel &phi_kernel = phi::KernelFactory::Instance().SelectKernel(pt_kernel_signature.name, pt_kernel_key);
  phi_kernel_ = &phi_kernel;
  bool is_valid = phi_kernel_->IsValid();
  PADDLE_ENFORCE_EQ(
        is_valid,
        true,
        platform::errors::Fatal(
            "%s phi kernel is invalid!.", pt_kernel_signature.name));
  
  // dev_ctx_.reset(new paddle::platform::CUDADeviceContext(place_));
  // paddle::platform::CUDADeviceContext *dev_ctx = new paddle::platform::CUDADeviceContext(place_);
  // std::shared_ptr<paddle::platform::CUDADeviceContext> dev_ctx(new paddle::platform::CUDADeviceContext(place_));
  // static paddle::platform::CUDADeviceContext dev_ctx(place_);
  
  dev_ctx_ = new paddle::platform::CUDADeviceContext(place_);
  dev_ctx_->SetAllocator(&allocator_);
  dev_ctx_->cublas_handle();
  dev_ctx_->cudnn_handle();

  phi_kernel_context_.reset(new phi::KernelContext(dev_ctx_));
  for(int i = 0; i < pt_kernel_signature.attr_names.size(); i++) {
    phi_kernel_context_->EmplaceBackAttr(Convert2PhiAttribute(op_desc_, pt_kernel_signature.attr_names[i]));
    LOG(INFO) << pt_kernel_signature.attr_names[i];
  }
  return 0; 
}

#if IS_TRT_VERSION_LT(8000)
  int GenericPlugin::enqueue(int batchSize,
                            const void* const* inputs,
                            void** outputs,
#else
  int GenericPlugin::enqueue(int batchSize,
                            const void* const* inputs,
                            void* const* outputs,
#endif
                            void* workspace,
                            cudaStream_t stream) TRT_NOEXCEPT {
  LOG(INFO) << "enqueue" << op_desc_.Type();
  std::vector<phi::DenseTensor>dense_tensor_inputs(input_dims_.size());
  std::vector<phi::DenseTensor>dense_tensor_outputs(output_dims_.size());

  // input
  for(int i = 0; i < input_dims_.size(); i++) {
    auto const &input_dims = this->getInputDims(i);
    float const *idata = reinterpret_cast<float const *>(inputs[i]);

    std::vector<int>input_shape;
    if(input_dims.nbDims == 3)
      input_shape.push_back(batchSize);
    for (int i = 0; i < input_dims.nbDims; i++)
      input_shape.push_back(input_dims.d[i]);

    int input_numel = 1;
    for (int i = 0; i < input_shape.size(); i++) {
      input_numel *= input_shape[i];
      LOG(INFO) << "input_shape[i]: " << input_shape[i];
    }

    phi::DenseTensorMeta input_meta(DataType::FLOAT32, phi::make_ddim(input_shape));
    std::shared_ptr<phi::Allocation> input_alloc(new phi::Allocation((void*)idata, input_numel*sizeof(float), place_));
    dense_tensor_inputs[i] = std::move(phi::DenseTensor(input_alloc, input_meta));
    phi_kernel_context_->EmplaceBackInput(&dense_tensor_inputs[i]);
  }
  
  // output
  for(int i = 0; i < output_dims_.size(); i++) {
    auto const &output_dims = this->getOutputDims(i);
    float *const *odata = reinterpret_cast<float *const *>(outputs[i]);

    std::vector<int>output_shape;
    for (int i = 0; i < output_dims.nbDims; i++)
      output_shape.push_back(output_dims.d[i]);

    int output_numel = 1;
    for (int i = 0; i < output_shape.size(); i++) {
      output_numel *= output_shape[i];
      LOG(INFO) << "output_shape[i]: " << output_shape[i];
    }

    phi::DenseTensorMeta output_meta(DataType::FLOAT32, phi::make_ddim(output_shape));
    std::shared_ptr<phi::Allocation> output_alloc(new phi::Allocation((void*)odata, output_numel*sizeof(float), place_));
    dense_tensor_outputs[i] = std::move(phi::DenseTensor(output_alloc, output_meta));
    phi_kernel_context_->EmplaceBackOutput(&dense_tensor_outputs[i]);
  }
  
  paddle::platform::stream::CUDAStream cs(stream, place_);
  dev_ctx_->SetCudaStream(&cs);
  
  (*phi_kernel_)(phi_kernel_context_.get());

  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle