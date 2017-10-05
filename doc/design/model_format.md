# Design Doc: Model Format

## Motivation

The model is the output of training process. One complete model consists of two parts, namely, the **topology** and the **parameters**. To support business deployment, we need to make the model format must be self-completed and do not expose any training source code.

As a result, In PaddlePaddle, the **topology** represents as a  [ProgramDesc](https://github.com/PaddlePaddle/Paddle/blob/1c0a4c901c9fc881d120249c703b15d1c50dae7d/doc/design/program.md), which describes the model structure. The **parameters** contain all the trainable weights in the model, we must support large size parameter, and high efficiency read/write for speed. 

## Implementation

The topology is saved as a plain text, in detail, a self-complete protobuf file. 

The parameters are saved as a binary file. As we all know, the protobuf message has the limits of [64M size](https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.io.coded_stream#CodedInputStream.SetTotalBytesLimit.details). So we design a particular format for tensor serialization, for speed we core dump the memory to disk and save the necessary information, such as the`dims`, `name` of the tensor, Even the `LoD` information in [LoDTensor](https://github.com/PaddlePaddle/Paddle/blob/1c0a4c901c9fc881d120249c703b15d1c50dae7d/paddle/framework/lod_tensor.md). In detail, as the table shows

```text
[offset] [type]          [value]          [description] 
0000     32 bit integer  1            	  1 for little endian, 0 for big endian
0004     32 bit integer  0x0000006E(0110) version number, 0110 for paddle version 0.11.0
0008     32 bit integer  ??               md5checksum of Tensors
0009     unsigned byte    0			   	 0 for tensor, 1 for LoDTensor
0013     32 bit integer  28               Tensor Name length
0017     unsigned byte   ??               Tensor Name chars 
0018     unsigned byte   ??               ..
...
00100     32 bit integer  3               Tensor dims count
00104     32 bit integer  ??              Tensor dims 0 
00108     32 bit integer  ??               ..
...
00150     unsigned byte   ??              Tensor value
00151     unsigned byte   ??               ..

```

## Summary

We introduce the model format, the `ProgramDesc` describe the **topology**, and a particular binary format for **parameters**.
