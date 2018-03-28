C-API Inference Library
========================

After we train a neural network, we use it to do inference. Inference is the process of preparing input data and propagating it through the model to produce the result.

Compared with model training, prediction has the following features:

#. Inference does not require backpropagation and parameter updates, as required during training.
#. Labels are not needed in prediction.
#. Most of the time, predictions need to be integrated with the user system.

Therefore, the model prediction SDK needs to be designed separately and has the following features:

#. The predictive SDK does not include backpropagation and parameter updates to reduce the size of the SDK.
#. The predictive SDK needs a simple user interface for ease of use.
#. Since the input data may have a variety of structures, the format of the input data is clearly and compactly packaged.
#. In order to be compatible with user's system, the SDK's interface must conform to the C-standard interface.

PaddlePaddle provides C-API to solve the above problem. Following are the guidelines to use the C-API:

..  toctree::
  :maxdepth: 1

  compile_paddle_lib_en.md
  organization_of_the_inputs_en.md
  workflow_of_capi_en.md
