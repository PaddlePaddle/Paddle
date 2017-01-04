#include "PaddleCAPI.h"
#include "PaddleCAPIPrivate.h"
#include "paddle/gserver/gradientmachines/NeuralNetwork.h"

#define cast(v) paddle::capi::cast<paddle::capi::CGradientMachine>(v)

enum GradientMatchineCreateMode {
  CREATE_MODE_NORMAL = 0,
  CREATE_MODE_TESTING = 4
};

namespace paddle {

class MyNeuralNetwork : public NeuralNetwork {
public:
  MyNeuralNetwork(const std::string& name, NeuralNetwork* network)
      : NeuralNetwork(name, network) {}
};

NeuralNetwork* newCustomNerualNetwork(const std::string& name,
                                      NeuralNetwork* network) {
  return new MyNeuralNetwork(name, network);
}
}

extern "C" {
int PDGradientMachineCreateForPredict(PD_GradiemtMachine* machine,
                                      void* modelConfigProtobuf,
                                      int size) {
  if (modelConfigProtobuf == nullptr) return PD_NULLPTR;
  paddle::ModelConfig config;
  if (!config.ParseFromArray(modelConfigProtobuf, size) ||
      !config.IsInitialized()) {
    return PD_PROTOBUF_ERROR;
  }

  auto ptr = new paddle::capi::CGradientMachine();
  ptr->machine.reset(paddle::GradientMachine::create(
      config, CREATE_MODE_TESTING, {paddle::PARAMETER_VALUE}));
  *machine = ptr;
  return PD_NO_ERROR;
}

int PDGradientMachineDestroy(PD_GradiemtMachine machine) {
  delete cast(machine);
  return PD_NO_ERROR;
}
}
