#pragma once 
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"
namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {
    class GroupNormPluginDynamic : public DynamicPluginTensorRT{
        public:
        GroupNormPluginDynamic(
            const float * scale,
            const int scale_num,
            const float * bias,
            const int bias_num,
            float eps,
            int groups,
            const std::string& data_layout_str,
            std::vector<int64_t> mean_shape,
            std::vector<int64_t> variance_shape)
        : groups_(groups),
          eps_(eps),
          data_layout_str_(data_layout_str),
          mean_shape_(mean_shape),
          variance_shape_(variance_shape) {
            bias_.resize(bias_num);
            scale_.resize(scale_num);
            std::copy(bias,bias+bias_num,bias_.data());
            std::copy(scale,scale+scale_num,scale_.data());
        }

        GroupNormPluginDynamic(void const * serialData, size_t serialLength){
            DeserializeValue(&serialData,&serialLength,&scale_);
            DeserializeValue(&serialData,&serialLength,&bias_);
            DeserializeValue(&serialData,&serialLength,&groups_);
            DeserializeValue(&serialData,&serialLength,&eps_);
            DeserializeValue(&serialData,&serialLength,&data_layout_str_);
            DeserializeValue(&serialData,&serialLength,&mean_shape_);
            DeserializeValue(&serialData,&serialLength,&variance_shape_);
        }
        nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
            return new GroupNormPluginDynamic(
                                              scale_.data(),
                                              scale_.size(),
                                              bias_.data(),
                                              bias_.size(),
                                              eps_,
                                              groups_,
                                              mean_shape_,
                                              variance_shape_);
        }

        const char* getPluginType() const TRT_NOEXCEPT override{
            return "groupnorm_plugin_dynamic";
        }
        int getNbOutputs() const TRT_NOEXCEPT override {return 1;}
        int initialize() TRT_NOEXCEPT override { return 0; }
        
        size_t getSerializationSize() const TRT_NOEXCEPT override {
            return SerializedSize(bias_)+SerializedSize(scale_)+
                   SerializedSize(groups_)+SerializedSize(eps_)+
                   SerializedSize(mean_shape_)+SerializedSize(variance_shape_);
        }
        void serialize(void * buffer) const TRT_NOEXCEPT override {
            SerializeValue(&buffer, bias_);
            SerializeValue(&buffer, scale_);
            SerializeValue(&buffer, groups_);
            SerializeValue(&buffer, eps_);
            SerializeValue(&buffer, mean_shape_);
            SerializeValue(&buffer, variance_shape_);
        }
        nvinfer1::DimsExprs getOutputDimensions(int output_index,
                                          const nvinfer1::DimsExprs* inputs,
                                          int nb_inputs,
                                          nvinfer1::IExprBuilder& expr_builder)
        TRT_NOEXCEPT override;

        bool supportsFormatCombination(int pos,
                                const nvinfer1::PluginTensorDesc* inOut,
                                int nbInputs,
                                int nbOutputs) TRT_NOEXCEPT override;
                                
        void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) TRT_NOEXCEPT override {}

        size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                                int nbInputs,
                                const nvinfer1::PluginTensorDesc* outputs,
                                int nbOutputs) const TRT_NOEXCEPT override {
        return 0;
        //TODO wang bojun return the appromix workspace layout need by plugin
        // a optim point Maybes 
        }
        int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                    const nvinfer1::PluginTensorDesc* outputDesc,
                    const void *const * inputs,
                    void* const* outputs,
                    void* workspace,
                    cudaStream_t stream) TRT_NOEXCEPT override;
        nvinfer1::DataType getOutputDataType(int index,
                                            const nvinfer1::DataType* inputTypes,
                                            int nbInputs) const
            TRT_NOEXCEPT override;

        void destroy() TRT_NOEXCEPT override { delete this; }

        private:
        std::vector<float> scale_;
        std::vector<float> bias_;
        framework::Tensor scale_t;
        framework::Tensor bias_t;
        framework::Tensor mean_t;
        framework::Tensor variance_t;
        int groups_;
        float eps_;
        std::vector<int64_t> mean_shape_;
        std::vector<int64_t> variance_shape_;
    };
    class GroupNormPluginDynamicCreater : public TensorRTPluginCreator{
        public:
        const char* getPluginName() const TRT_NOEXCEPT override {
            return "groupnorm_plugin_dynamic";
        }
        const char* getPluginVersion() const TRT_NOEXCEPT override{
            return "1";
        }
        nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                               const void* serial_data,
                                               size_t serial_length) TRT_NOEXCEPT override 
            {
                return new GroupNormPluginDynamic(serial_data,serial_length);
            }
    };
    REGISTER_TRT_PLUGIN_V2(GroupNormPluginDynamicCreater);
}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

