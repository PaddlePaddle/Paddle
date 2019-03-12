#include "paddle/fluid/inference/op_lite/concat_op.h"
#include <Eigen/Eigen>
#include <vector>
#include "paddle/fluid/inference/op_lite/op_lite.h"
#include "paddle/fluid/operators/concat_op.h"

namespace paddle {
namespace inference {
namespace op_lite {

bool Concat::CheckShape() const {
  CHECK_GT_OR_FALSE(param_.inputs.size(), 0);
  CHECK_OR_FALSE(param_.output);
  
  return true;
}

bool Concat::InferShape() const {
  VLOG(3) << "In concat_op stage: InferShape.";
  VLOG(3) << "In concat_op inputs size: " << param_.inputs.size();
  VLOG(3) << "In concat_op inputs front: " << param_.inputs.front();
  auto out_dims = param_.inputs.front()->dims();
  size_t in_zero_dims_size = out_dims.size();
  VLOG(3) << "In concat_op in_zero_dims_size: " << in_zero_dims_size;
  const size_t n = param_.inputs.size();
  VLOG(3) << "In concat_op n: " << n;
  for (size_t i = 1; i < n; ++i) {
    for (size_t j = 0; j < in_zero_dims_size; ++j) {
      if (j == param_.axis) {
        out_dims[param_.axis] += param_.inputs[i]->dims()[j];
      } else {
        CHECK_EQ_OR_FALSE(out_dims[j], param_.inputs[i]->dims()[j]);
      }
      VLOG(3) << "In concat_op out_dims[" << j << "] " << out_dims[j];
    }
  }
  if (out_dims[param_.axis] < 0) {
    out_dims[param_.axis] = -1;
  }
  param_.output->Resize(out_dims);
  // share lod how 
  // TODO (Jane) don't understand
  param_.output->set_lod(param_.inputs.front()->lod());
  return true;
}

bool Concat::Run() {
  using T = float;
  if (param_.axis == 0 && param_.inputs.size() < 10) {
    size_t output_offset = 0;
    for (auto* in : param_.inputs) {
      auto in_stride = framework::stride_numel(in->dims());
      auto out_stride = framework::stride_numel(param_.output->dims());
      operators::StridedNumelCopyWithAxis<float>(platform::CPUDeviceContext(), param_.axis,
                                     param_.output->data<float>() + output_offset, out_stride,
                                     in->data<float>(), in_stride, in_stride[param_.axis]);
      output_offset += in_stride[param_.axis];
    }
  } else {
    size_t input_size = param_.inputs.size();
    VLOG(3) << "Concat Run input_size: " << input_size;
    std::vector<framework::Tensor> inputs(input_size);
    for (size_t j = 0; j < input_size; ++j) {
      inputs[j] = *(param_.inputs[j]);
      VLOG(3) << "Concat Run copy " << j << " value " << inputs[j].memory_size();
    }
    param_.output->mutable_data<T>(platform::CPUPlace());
    VLOG(3) << "Concat Run output " << param_.output << " " << param_.output->memory_size();
    paddle::operators::math::ConcatFunctor<platform::CPUDeviceContext, float> concat_functor;
    concat_functor(platform::CPUDeviceContext(), inputs,
                   static_cast<int>(param_.axis), param_.output);
  }  
  return true;
}

bool Concat::Build(const paddle::framework::OpDesc &opdesc,
                 framework::Scope *scope) {
  const auto &inputs = opdesc.Inputs();
  const auto &outputs = opdesc.Outputs();
  CHECK_OR_FALSE(inputs.count("X"));
  CHECK_OR_FALSE(outputs.count("Out"));

  param_.axis = boost::get<int>(opdesc.GetAttr("axis"));
  param_.inputs.clear();

  VLOG(4) << "concat attr: " << param_.axis;

  for (auto& name : inputs.at("X")) {
    VLOG(4) << "concat build input name: " << name;
    auto x = scope->FindVar(name);
    param_.inputs.push_back(x->GetMutable<LoDTensor>());
  }
  auto out = scope->FindVar(outputs.at("Out").front());
  param_.output = out->GetMutable<LoDTensor>();
  return true;
}

std::string Concat::DebugString() const { return "concat-lite-op"; }

}  // namespace op_lite
}  // namespace inference
}  // namespace paddle
