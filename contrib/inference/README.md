# Embed Paddle Inference in Your Application

Paddle inference offers the APIs in `C` and `C++` languages.

One can easily deploy a model trained by Paddle following the steps as below:

1. Optimize the native model;
2. Write some codes for deployment.


Let's explain the steps in detail.

## Optimize the native Fluid Model

The native model that get from the training phase needs to be optimized for that.

- Clean the noise such as the cost operators that do not need inference;
- Prune unnecessary computation fork that has nothing to do with the output;
- Remove extraneous variables;
- Memory reuse for native Fluid executor;
- Translate the model storage format to some third-party engine's, so that the inference API can utilize the engine for acceleration;

We have an official tool to do the optimization, call `paddle_inference_optimize --help` for more information.

## Write some codes

Read `paddle_inference_api.h` for more information.
