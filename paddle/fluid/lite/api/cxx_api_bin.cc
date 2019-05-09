#include "paddle/fluid/lite/api/cxx_api.h"
#include "paddle/fluid/lite/core/mir/passes.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {

void Run(const char* model_dir) {
  lite::LightPredictor predictor;
#ifndef LITE_WITH_CUDA
  std::vector<Place> valid_places({Place{TARGET(kHost), PRECISION(kFloat)}});
#else
  std::vector<Place> valid_places({
      Place{TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW)},
      Place{TARGET(kCUDA), PRECISION(kFloat), DATALAYOUT(kNCHW)},
      Place{TARGET(kCUDA), PRECISION(kAny), DATALAYOUT(kNCHW)},
      Place{TARGET(kHost), PRECISION(kAny), DATALAYOUT(kNCHW)},
      Place{TARGET(kCUDA), PRECISION(kAny), DATALAYOUT(kAny)},
      Place{TARGET(kHost), PRECISION(kAny), DATALAYOUT(kAny)},
  });
#endif

  predictor.Build(model_dir, Place{TARGET(kCUDA), PRECISION(kFloat)},
                  valid_places);

  auto* input_tensor = predictor.GetInput(0);
  input_tensor->Resize(DDim(std::vector<DDim::value_type>({100, 100})));
  auto* data = input_tensor->mutable_data<float>();
  for (int i = 0; i < 100 * 100; i++) {
    data[i] = i;
  }

  LOG(INFO) << "input " << *input_tensor;

  predictor.Run();

  auto* out = predictor.GetOutput(0);
  LOG(INFO) << out << " memory size " << out->data_size();
  LOG(INFO) << "out " << out->data<float>()[0];
  LOG(INFO) << "out " << out->data<float>()[1];
  LOG(INFO) << "dims " << out->dims();
  LOG(INFO) << "out " << *out;
}

}  // namespace lite
}  // namespace paddle

int main(int argc, char** argv ) {
  CHECK_EQ(argc, 2) << "usage: ./cmd <model_dir>";
  paddle::lite::Run(argv[1]);

  return 0;
}


USE_LITE_OP(mul);
USE_LITE_OP(fc);
USE_LITE_OP(scale);
USE_LITE_OP(feed);
USE_LITE_OP(fetch);
USE_LITE_OP(io_copy);
USE_LITE_KERNEL(fc, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(mul, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(scale, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(feed, kHost, kAny, kAny, def);
USE_LITE_KERNEL(fetch, kHost, kAny, kAny, def);

#ifdef LITE_WITH_CUDA
USE_LITE_KERNEL(mul, kCUDA, kFloat, kNCHW, def);
USE_LITE_KERNEL(io_copy, kCUDA, kAny, kAny, host_to_device);
USE_LITE_KERNEL(io_copy, kCUDA, kAny, kAny, device_to_host);
#endif
