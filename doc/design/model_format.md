# Design Doc: Model Format

## Motivation

The model is the output of training process. One complete model consists of two parts, namely, the **topology** and the **parameters**. To support industrial deployment, we need to make the model format must be self-completed and do not expose any training source code.

As a result, In PaddlePaddle, the **topology** represents as a  [ProgramDesc](https://github.com/PaddlePaddle/Paddle/blob/1c0a4c901c9fc881d120249c703b15d1c50dae7d/doc/design/program.md), which describes the model structure. The **parameters** contain all the trainable weights in the model, we must support large size parameter, and efficient serialization/deserialization. 

## Implementation

The topology is saved as a plain text, in detail, a self-contain protobuf file. 

The parameters are saved as a binary file. As we all know, the protobuf message has the limits of [64M size](https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.io.coded_stream#CodedInputStream.SetTotalBytesLimit.details). We do a (benchmark experiment)[https://github.com/PaddlePaddle/Paddle/pull/4610], its result shows protobuf is not fit in this scene.

As a result, we design a particular format for tensor serialization. By default, arbitrary tensor in Paddle is a [LoDTensor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/lod_tensor.md), and has a description information proto of [LoDTensorDesc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/framework.proto#L99). We save the DescProto as the byte string header, it contains the necessary information, such as the `dims`, and the `LoD` information in [LoDTensor](https://github.com/PaddlePaddle/Paddle/blob/1c0a4c901c9fc881d120249c703b15d1c50dae7d/paddle/framework/lod_tensor.md). Tensor stores value in a continuous memory buffer, for speed we dump the raw memory to disk and save it as the byte string content. So, the binary format of one tensor is, 

In detail, tensor's  byte view as the table shows. Note that all the signed value written in little-endian.

|field name  | type | description |
| --- | --- | --- |
| version | uint32_t | Version of saved file. Always 0 now. |
| tensor desc length | uint32_t | TensorDesc(Protobuf message) length in bytes. |
| tensor desc | void* | TensorDesc protobuf binary message |
| tensor data | void* | Tensor's data in binary format. The length of `tensor_data` is decided by `TensorDesc.dims()` and `TensorDesc.data_type()` |
| lod_level | uint64_t | Level of LoD |
| length of lod[0] | uint64_t | [Optional] length of lod[0] in bytes. |
| data of lod[0] | uint64_t*  | [Optional] lod[0].data() |
| ... | ... | ... |



## Summary

We introduce the model format, the `ProgramDesc` describe the **topology**, and a bunch of particular format binary tensors describes the **parameters**.
