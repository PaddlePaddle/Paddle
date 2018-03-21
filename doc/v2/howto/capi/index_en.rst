C-API Prediction Library
========================

After we trained a neural network model, then we use it to make predictions. Prediction is a process of preparing input data and processing the model to get the result.

Compared with model training, prediction has the following features:

#. Prediction does not require backpropagation and parameter updates during training.
#. Labels are not needed in prediction.
#. Most of the time, predictions need to be integrated with the user system.

Therefore, the model prediction SDK needs to be designed separately and has the following features:

#. The predictive SDK does not include reverse propagation and parameter updates to reduce the size of the SDK.
#. The predictive SDK needs a simple user interface for ease of use.
#. Since the input data may have a variety of structures, the format of the input data is clearly and compactly packaged.
#. In order to be compatible with the user's system, the SDK's interface must conform to the C-standard interface.

PaddlePaddle provides C-API to solve the above problem. The use of C-API, we provide the following guidelines:

..  toctree::
  :maxdepth: 1

  compile_paddle_lib_en.md
  organization_of_the_inputs_en.md
  workflow_of_capi_en.md
