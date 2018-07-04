# Design Doc: Model Format

## Motivation

A model is an output of the training process. One complete model consists of two parts, the **topology** and the **parameters**. In order to support industrial deployment, the model format must be self-complete and must not expose any training source code.

As a result, In PaddlePaddle, the **topology** is represented as a  [ProgramDesc](https://github.com/PaddlePaddle/Paddle/blob/1c0a4c901c9fc881d120249c703b15d1c50dae7d/doc/design/program.md), which describes the model structure. The **parameters** contain all the trainable weights in the model. We must support large size parameters and efficient serialization/deserialization of parameters.

## Implementation

The topology is saved as a plain text in a detailed self-contain protobuf file.

The parameters are saved as a binary file. As we all know, the protobuf message has a limit of [64M size](https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.io.coded_stream#CodedInputStream.SetTotalBytesLimit.details). We have done a [benchmark experiment](https://github.com/PaddlePaddle/Paddle/pull/4610), which shows that protobuf is not fit for the task.

As a result, we design a particular format for tensor serialization. By default, an arbitrary tensor in Paddle is a [LoDTensor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/lod_tensor.md), and has a description information proto of [LoDTensorDesc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/framework.proto#L99). We save the DescProto as the byte string header. It contains all the necessary information, such as the `dims`, and the `LoD` information in [LoDTensor](https://github.com/PaddlePaddle/Paddle/blob/1c0a4c901c9fc881d120249c703b15d1c50dae7d/paddle/framework/lod_tensor.md). A tensor stores values in a continuous memory buffer. For speed we dump the raw memory to disk and save it as the byte string content. So, the binary format of one tensor is,

The table below shows a tensor's byte view in detail. Note that all the signed values are written in the little-endian format.

<table>
<thead>
<tr>
<th>field name</th>
<th>type </th>
<th>description </th>
</tr>
</thead>
<tbody>
<tr>
<td> version</td>
<td> uint32_t </td>
<td> Version of saved file. Always 0 now.</td>
</tr>

<tr>
<td> tensor desc length  </td>
<td> uint32_t </td>
<td> TensorDesc(Protobuf message) length in bytes. </td>
</tr>
<tr>
<td>tensor desc </td>
<td> void*</td>
<td> TensorDesc protobuf binary message </td>
</tr>
<tr>
<td> tensor data </td>
<td> void* </td>
<td> Tensor's data in binary format. The length of `tensor_data` is decided by `TensorDesc.dims()` and `TensorDesc.data_type()` </td>
</tr>
<tr>
<td> lod_level</td>
<td> uint64_t </td>
<td> Level of LoD </td>
</tr>
<tr>
<td> length of lod[0] </td>
<td> uint64_t </td>
<td> [Optional] length of lod[0] in bytes. </td>
</tr>
<tr>
<td> data of lod[0] </td>
<td> uint64_t*   </td>
<td> [Optional] lod[0].data() </td>
</tr>
<tr>
<td>... </td>
<td> ... </td>
<td> ... </td>
</tr>
</tbody>
</table>

## Summary

- We introduce a model format.
- The model represented by its forward-pass computation procedure is saved in a **ProgramDesc** protobuf message.
- A bunch of specified format binary tensors describe the **parameters**.
