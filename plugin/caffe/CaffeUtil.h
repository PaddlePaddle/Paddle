/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include <caffe/proto/caffe.pb.h>
#include <caffe/layer.hpp>
#include <caffe/blob.hpp>
#include <caffe/layer_factory.hpp>
#include <caffe/common.hpp>

namespace paddle {

bool ReadProtoFromTextContent(const std::string& text,
                              ::google::protobuf::Message* proto) {
  bool success = google::protobuf::TextFormat::ParseFromString(text, proto);
  return success;
}

inline caffe::LayerParameter* getLayerParameter(const std::string& value) {
  caffe::NetParameter net_param;
  if (!ReadProtoFromTextContent(value, &net_param))
    CHECK(false) << "Caffe Net Prototxt: " << value << "Initialized Failed";

  CHECK_EQ(net_param.layer_size(), 1) << "Protoxt " << value
                                      << " is more than one layer";
  return new caffe::LayerParameter(net_param.layer(0));
}

void setMode(bool useGpu) {
  if (useGpu) {
    ::caffe::Caffe::set_mode(::caffe::Caffe::GPU);
  } else {
    ::caffe::Caffe::set_mode(::caffe::Caffe::CPU);
  }
}

template <typename Dtype>
class LayerRegistry {
public:
  static ::caffe::Layer<Dtype>* CreateLayer(
      const ::caffe::LayerParameter& param) {
    ::caffe::shared_ptr<::caffe::Layer<Dtype>> ptr =
        ::caffe::LayerRegistry<Dtype>::CreateLayer(param);
    // avoid caffe::layer destructor, which deletes the weights layer owns
    new ::caffe::shared_ptr<::caffe::Layer<Dtype>>(ptr);
    return ptr.get();
  }
};
}
