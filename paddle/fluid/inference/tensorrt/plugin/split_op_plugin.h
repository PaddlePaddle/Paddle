
#pragma once

#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"
#include <thrust/device_vector.h>

namespace paddle {
namespace inference {
namespace tensorrt {

class SplitPlugin : public PluginTensorRT {
  int axis_;
  std::vector<int> output_lenght_;
  int nx_, ny_, nz_;
  thrust::device_vector<int> d_segment_offsets_;

 protected:
  virtual size_t getSerializationSize() override {
    return serialized_size(axis_) + serialized_size(output_lenght_)
      + getBaseSerializationSize();
  }

  virtual void serialize(void *buffer) override {
    serializeBase(buffer);
    serialize_value(&buffer, axis_);
    serialize_value(&buffer, output_lenght_);
  }

 public:
  Split() {}
  SplitPlugin(void const* serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    deserialize_value(&serialData, &serialLength, &axis_);
    deserialize_value(&serialData, &serialLength, &output_lenght_);
  }

  SplitPlugin* clone() const override {
    return new SplitPlugin(axis_, output_lenght_);
  }

  virtual const char* getPluginType() const override { return "split"; }
  virtual int getNbOutputs() const override { return output_lenght_.size(); }
  virtual nvinfer1::Dims getOutputDimensions(int index,
                                             const nvinfer1::Dims *inputs, int nbInputDims) override;
  virtual int initialize() override;
  virtual int enqueue(int batchSize,
                      const void *const *inputs, void **outputs,
                      void *workspace, cudaStream_t stream) override;

  void setAxis(int axis) {
    axis_ = axis;
  }

  void setOutputLengths(const std::vector<int> & output_lengths) {
    output_length_ = output_lengths;
  }

};

} // tensorrt
} // inference
} // paddle
