#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/compat/op_utils.h"
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/inference/tensorrt/plugin/generic_dynamic_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

platform::CUDAPlace GenericDynamicPlugin::place_;
// paddle::platform::CUDADeviceContext GenericPlugin::dev_ctx_(GenericPlugin::place_);
paddle::memory::allocation::CUDAAllocator GenericDynamicPlugin::allocator_(place_);

inline std::string DimsToStr(nvinfer1::Dims dims) {
  std::stringstream ss;
  for(size_t i = 0; i < dims.nbDims; i++) {
    ss << dims.d[i] << " ";
  }
  return ss.str();
}

int GenericDynamicPlugin::initialize() TRT_NOEXCEPT { 
  // LOG(INFO) << "initialize " << op_desc_.Type();
  proto_op_desc_.ParseFromString(op_meta_data_);
  op_desc_ = std::move(framework::OpDesc(proto_op_desc_, nullptr));
  std::string op_type = op_desc_.Type();
  // return 0;
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
  phi_kernel_ = new phi::Kernel(phi_kernel);
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
  
  phi_kernel_context_.reset(new phi::KernelContext(dev_ctx_));
  CHECK(phi_kernel_context_);
  for(int i = 0; i < pt_kernel_signature.attr_names.size(); i++) {
    // LOG(INFO) << pt_kernel_signature.attr_names[i];
    phi_kernel_context_->EmplaceBackAttr(Convert2PhiAttribute(op_desc_, pt_kernel_signature.attr_names[i]));
  }
  return 0; 
}

nvinfer1::DimsExprs GenericDynamicPlugin::getOutputDimensions(
      int output_index,
      const nvinfer1::DimsExprs* inputs,
      int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT{
  PADDLE_ENFORCE_EQ(nb_inputs,
                    getNbInputs(),
                    platform::errors::InvalidArgument(
                        "The %s plugin should be %d input.", op_desc_.Type().c_str(), getNbInputs()));
  if(op_desc_.Type() == "roll")
    return inputs[0];
  else if(op_desc_.Type() == "index_select") {
    nvinfer1::DimsExprs output(inputs[0]);

    int dim = PADDLE_GET_CONST(int, op_desc_.GetAttr("dim"));
    output.d[dim] = expr_builder.constant(inputs[1].d[0]->getConstantValue());
    return output;
  } 
  else if(op_desc_.Type() == "pool2d") {
    bool is_global_ = PADDLE_GET_CONST(bool, op_desc_.GetAttr("global_pooling"));
    bool adaptive_ = PADDLE_GET_CONST(bool, op_desc_.GetAttr("adaptive"));

    // std::string pool_type = PADDLE_GET_CONST(std::string, op_desc.GetAttr("pooling_type"));
    std::vector<int> ksize_ = PADDLE_GET_CONST(std::vector<int>, op_desc_.GetAttr("ksize"));
    std::vector<int> strides_ = PADDLE_GET_CONST(std::vector<int>, op_desc_.GetAttr("strides"));
    std::vector<int> paddings_ = PADDLE_GET_CONST(std::vector<int>, op_desc_.GetAttr("paddings"));
    bool ceil_mode_ = PADDLE_GET_CONST(bool, op_desc_.GetAttr("ceil_mode"));

    nvinfer1::DimsExprs output(inputs[0]);
    if (is_global_ && !adaptive_) {
      output.d[2] = expr_builder.constant(1);
      output.d[3] = expr_builder.constant(1);
      return output;
    }
    if (is_global_ && adaptive_) {
      return inputs[0];
    }
    if (adaptive_) {
      output.d[2] = expr_builder.constant(ksize_[0]);
      output.d[3] = expr_builder.constant(ksize_[1]);
      return output;
    }

    auto stri_0 = expr_builder.constant(strides_[0]);
    auto stri_1 = expr_builder.constant(strides_[1]);
    auto one_value = expr_builder.constant(1);

    auto v0_tmp = expr_builder.constant(-ksize_[0] + 2 * paddings_[0]);
    auto v1_tmp = expr_builder.constant(-ksize_[1] + 2 * paddings_[1]);

    auto ceil_tmp =
        expr_builder.constant(-ksize_[0] + 2 * paddings_[0] + strides_[0] - 1);
    auto ceil1_tmp =
        expr_builder.constant(-ksize_[1] + 2 * paddings_[1] + strides_[1] - 1);

    if (!ceil_mode_) {
      output.d[2] = expr_builder.operation(
          nvinfer1::DimensionOperation::kSUM,
          *expr_builder.operation(
              nvinfer1::DimensionOperation::kFLOOR_DIV,
              *expr_builder.operation(
                  nvinfer1::DimensionOperation::kSUM, *inputs[0].d[2], *v0_tmp),
              *stri_0),
          *one_value);
      output.d[3] = expr_builder.operation(
          nvinfer1::DimensionOperation::kSUM,
          *expr_builder.operation(
              nvinfer1::DimensionOperation::kFLOOR_DIV,
              *expr_builder.operation(
                  nvinfer1::DimensionOperation::kSUM, *inputs[0].d[3], *v1_tmp),
              *stri_1),
          *one_value);

    } else {
      output.d[2] = expr_builder.operation(
          nvinfer1::DimensionOperation::kSUM,
          *expr_builder.operation(
              nvinfer1::DimensionOperation::kFLOOR_DIV,
              *expr_builder.operation(
                  nvinfer1::DimensionOperation::kSUM, *inputs[0].d[2], *ceil_tmp),
              *stri_0),
          *one_value);
      output.d[3] = expr_builder.operation(
          nvinfer1::DimensionOperation::kSUM,
          *expr_builder.operation(
              nvinfer1::DimensionOperation::kFLOOR_DIV,
              *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                      *inputs[0].d[3],
                                      *ceil1_tmp),
              *stri_1),
          *one_value);
    }
    return output;
  } 
  else if(op_desc_.Type() == "slice") {

  }
  else {
    LOG(ERROR) << op_desc_.Type() << " getOutputDimensions has no implement";
  }
}

void GenericDynamicPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                               int nb_inputs,
                               const nvinfer1::DynamicPluginTensorDesc* out,
                               int nb_outputs) TRT_NOEXCEPT{
  // CHECK(phi_kernel_context_); 
}

int GenericDynamicPlugin::enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                      const nvinfer1::PluginTensorDesc* output_desc,
                      const void* const* inputs,
                      void* const* outputs,
                      void* workspace,
                      cudaStream_t stream) TRT_NOEXCEPT{
  // Consider only input, output
  // LOG(INFO) << "enqueue " << op_desc_.Type();
  std::vector<phi::DenseTensor> dense_tensor_inputs(getNbInputs());
  std::vector<phi::DenseTensor> dense_tensor_outputs(getNbOutputs());

  // weight (temporarily not consider)
  
  // input
  for(int i = 0; i < getNbInputs(); i++) {
    auto const &input_dims = input_desc[i].dims;

    std::vector<int>input_shape;
    for (int j = 0; j < input_dims.nbDims; j++)
      input_shape.push_back(input_dims.d[j]);

    int input_numel = 1;
    for (int k = 0; k < input_shape.size(); k++) {
      input_numel *= input_shape[k];
      // LOG(INFO) << "input_shape[k]: " << input_shape[k];
    }
    // temporary code for index_select op
    // if(i == 1) {
    //   phi::DenseTensorMeta input_meta(DataType::INT64, phi::make_ddim(input_shape));
    //   std::shared_ptr<phi::Allocation> input_alloc(new phi::Allocation((void*)(inputs[i]), input_numel*sizeof(int64_t), place_));
    //   dense_tensor_inputs[i] = std::move(phi::DenseTensor(input_alloc, input_meta));
    //   phi_kernel_context_->EmplaceBackInput(&dense_tensor_inputs[i]);
    // }
    // else {
    phi::DenseTensorMeta input_meta(DataType::FLOAT32, phi::make_ddim(input_shape));
    std::shared_ptr<phi::Allocation> input_alloc(new phi::Allocation((void*)(inputs[i]), input_numel*sizeof(float), place_));
    dense_tensor_inputs[i] = std::move(phi::DenseTensor(input_alloc, input_meta));
    phi_kernel_context_->EmplaceBackInput(&dense_tensor_inputs[i]);
    // }
  }  

  // output
  for(int i = 0; i < getNbOutputs(); i++) {
    auto const &output_dims = output_desc[i].dims;

    std::vector<int>output_shape;
    for (int j = 0; j < output_dims.nbDims; j++)
      output_shape.push_back(output_dims.d[j]);

    int output_numel = 1;
    for (int k = 0; k < output_shape.size(); k++) {
      output_numel *= output_shape[k];
      // LOG(INFO) << "output_shape[i]: " << output_shape[k];
    }
    
    //we should consider DataType， it not always is DataType::FLOAT32
    phi::DenseTensorMeta output_meta(DataType::FLOAT32, phi::make_ddim(output_shape));
    std::shared_ptr<phi::Allocation> output_alloc(new phi::Allocation((void*)(outputs[i]), output_numel*sizeof(float), place_));
    dense_tensor_outputs[i] = std::move(phi::DenseTensor(output_alloc, output_meta));
    phi_kernel_context_->EmplaceBackOutput(&dense_tensor_outputs[i]);
  }

  cs_ = new paddle::platform::stream::CUDAStream(stream, place_);
  dev_ctx_->SetCudaStream(cs_);

  (*phi_kernel_)(phi_kernel_context_.get());
  // cudaStreamSynchronize(stream);
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle