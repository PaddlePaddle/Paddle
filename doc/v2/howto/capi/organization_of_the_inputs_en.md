## Input/Output Data Organization

This article describes how to organize input data and use the feed-forward computation result of neural networks when using PaddlePaddle C-API.

### Input/Output Data Type

In terms of the implement of PaddlePaddle, the types of data input can be generally divided as:

1. One-dimensional array of integers.
2. Two-dimensional matrix of floats, including
    - dense matrix 
    - sparse matrix

Especially, 
1. The array type **can only store integers**, in order to support some scenarios like
    - word_id in natrual language processing.
    - class_id in classification tasks.
2. Tensors that have more than 2 dimensions, for example images with multi-channels or videos, can be represented as 2D matrix in coding. Users can follow the convenient way to represent tensors using 2D matrixs, we highly recommend users do this by themselves.
3. 2D-matrixes can represent both row vectors and column vectors, users need to use 2D-matrixes in order to represent float arrays.
4. For both array and matrix, **appending sequential information will transform them as sequential input. PaddlePaddle rely on those appended sequential information to judge whether an array/matrix is a kind of sequential information.** We will dscribe the definition of "sequential information" in the following sections.

### Basic Concepts for Usage

- In the inner-implment of PaddlePaddle, for each layer of a neural network, both the input and output data are organized as a `Argument` structure. If a layer has more than one input/output data, then each input/output will have its own `Argument`.
- `Argument` does not really "store" data, it just organically organizes input/output information together.
- As to the implement of `Argument`, arrays and matrixes are stored by `IVector` and `Matrix` store respectively. `Sequence Start Positions` (will be described in the following sections) contains the sequential information of input/outout data.

-**Notice that**:
    1. We will refer to the input/output data of each layer of PaddlePaddle as `argument` hereafter.
    2. We will refer to array in paddle as `paddle_ivector`.
    3. We will refer to matrix in paddle as `paddle_matrix`.

### Organzing Input/Output Data
- One-dimensional integer array
    `paddle_ivector`, which stands for integer array, usually can be used to represent discrete information, like class ids or word ids in natural language processing. The example below shows a `paddle_ivector` that has three elements, i.e., `1`, `2` and `3`:
    ```c
    int ids[] = {1, 2, 3};
     paddle_ivector ids_array =
         paddle_ivector_create(ids, sizeof(ids) / sizeof(int), false, false);
     CHECK(paddle_arguments_set_ids(in_args, 0, ids_array));
    ```
- **Dense Matrix**
    - A `m*n` matrix is consisted of `m` rows and `n` columns, each element is a float. To a neural network, the number `m` denotes batch-size, while `n` is the `size` of `paddle.layer.data`, which is configured when building the neural network.
    - The code below shows a `1 * layer_size` dense matrix, and each element is ramdomized.
    ```c
    paddle_matrix mat = paddle_matrix_create(
                            /* height = batch size */ 1,
                            /* width = dimensionality of the data layer */ layer_size,
                            /* whether to use GPU */ false);

    paddle_real* array;
    // Get the pointer pointing to the start address of the first row of the
    // created matrix.
    CHECK(paddle_matrix_get_row(mat, 0, &array));

    // Fill the matrix with a randomly generated test sample.
    srand(time(0));
    for (int i = 0; i < layer_size; ++i) {
      array[i] = rand() / ((float)RAND_MAX);
    }

    // Assign the matrix to the argument.
    CHECK(paddle_arguments_set_value(in_args, 0, mat));
    ```

- **Sparse Matrix**
  PaddlePaddle uses [CSR（Compressed Sparse Row Format）](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)) to store sparse matrix. The Figure-1 illustrates CSR.
  <p align="center">
  <img src="https://user-images.githubusercontent.com/5842774/34159369-009fd328-e504-11e7-9e08-36bc6dc5e505.png" width=700><br> Figure-1. Compressed Sparse Row Format. 
  </p>

CSR uses (1) values of non-zero elements (`values` in Figure-1); (2) row-offset (`row offsets` in Figure-1), which is the offset to the start position in a row. (3) indices of non-zero elements (`column indices` in Figure-1) to represent sparse matrix.

  In the C-API of PaddlePaddle, users can create sparse matrixes using the following API:
  ```c
  PD_API paddle_matrix paddle_matrix_create_sparse(
      uint64_t height, uint64_t width, uint64_t nnz, bool isBinary, bool useGpu);
  ```

  1. When creating sparse matrixes, some information needs to be configured: (1) height (batch-size). (2) width (`size` of `paddle.layer.data`), (3) number of non-zero elements ('nnz'). 
  2. When `isBinary` is set as `true`, **users only need to set `row_offset` and `colum indices`, excluding `values`**, and value of each element configured by `row_offset` and `column indices` is set as 1 by default.
  The following example shows a binary sparse matrix created on CPU:
  
  ```c
  paddle_matrix mat = paddle_matrix_create_sparse(1, layer_size, nnz, true, false);
  int colIndices[] = {9, 93, 109};  // layer_size here is greater than 109.
  int rowOffset[] = {0, sizeof(colIndices) / sizeof(int)};

  CHECK(paddle_matrix_sparse_copy_from(mat,
                                 rowOffset,
                                 sizeof(rowOffset) / sizeof(int),
                                 colIndices,
                                 (colIndices) / sizeof(int),
                                 NULL /*values array is NULL.*/,
                                 0 /*size of the value arrary is 0.*/));
  CHECK(paddle_arguments_set_value(in_args, 0, mat));
  ```
  The following example shows a sparse matrix created on CPU:
  ```c
  paddle_matrix mat = paddle_matrix_create_sparse(1, layer_size, nnz, false, false);
  int colIndices[] = {9, 93, 109};  // layer_size here is greater than 109.
  int rowOffset[] = {0, sizeof(colIndices) / sizeof(int)};
  float values[] = {0.5, 0.5, 0.5};

  CHECK(paddle_matrix_sparse_copy_from(mat,
                                 rowOffset,
                                 sizeof(rowOffset) / sizeof(int),
                                 colIndices,
                                 sizeof(colIndices) / sizeof(int),
                                 values,
                                 sizeof(values) / sizeof(float)));
  ```
  Notice that：
  1. Prediction on mobile devices **do not** support sparse matrix right now.

### Organizing Sequential Information

A sequence is consisted of multiple elements, including integers, floats, or vectors, with thier order information. Different sequences may have different numbers of elements. In PaddlePaddle, sequential input/output data are **integer arrays or float matrixes with sequential information** as described above. We now discuss in detail the definition of "sequential information".

A `batch` is a set of instances, which is fed to a neural network at once for feed-forward computation, the offset of each sequence in the `batch`, is the **sequential information** in PaddlePaddle, we name it "sequence start positions". Two kinds of sequential inforamtion is supported in PaddlePaddle:  

1. single-layer sequence
    - each element in the sequence is an essential computation unit, not a sequential data.
    - for example, a natural language sentence is a sequence, its elements are words.

2. double-layer sequence
    - Each element in the sequence is a sequence.
    - for example, a natural language paragraph is a double-layer sequence, paragraph is consisted of sentences and sentence is consisted of words.
    - double-layer sequence could help modeling long-term sequences or building multi-layer neural networks.

We refer to the sequential information of input/output data in PaddlePaddle as `sequence_start_positions` hereafter.

As for double-layer sequence, not only the offset information of each sequence shall be provided, but also the offset of sequences over the whole `batch` is needed. In other words, **double-layer sequence needs to be configured the `sequence_start_positions` for both the inner-sequence and the outer-sequence**.

**Notice that**
1. `sequence_start_poistions` represents the `offset of elements`, not the bytes that the data actually takes.
2. non-sequential input shall not be configured with `sequence_start_positions`.
3. **both single-layer and double-layer sequences are based on `paddle_ivector` in the implement of PaddlePaddle**.

Figure-2 illustrates how single-layer and double-layer sequences are stored in PaddlePaddle. 

<p align="center">
<img src="https://user-images.githubusercontent.com/5842774/34159714-1f81a9be-e505-11e7-8a8a-4902146ec899.png" width=800><br>Figure-2. Illustration of sequential input data. 
</p>

- single-layer sequence
    Figure-2 (a) shows a `batch` of 4 sequences:
    1. the length of each sequence is: 5, 3, 2, 4.
    2. the `sequence_start_positions` is `[0, 5, 8, 10, 14]`
    3. while training on local devices, both `paddle_ivector` and `paddle_matrix` can be appended with sequential information using the following API, making them as single-layer sequences:

    ```c
    int seq_pos_array[] = {0, 5, 8, 10, 14};
    paddle_ivector seq_pos = paddle_ivector_create(
        seq_pos_array, sizeof(seq_pos_array) / sizeof(int), false, false);
    // Suppose the network only has one input data layer.
    CHECK(paddle_arguments_set_sequence_start_pos(in_args, 0, 0, seq_pos));
    ```

- double-layer sequence
    Figure-2 (b) shows a `batch` of 4 sequences:
    1. the length of each sequence is: 5, 3, 2, 4. While each sequences has 3, 2, 1, 2 sub-sequences.
    2. the sequential information need to be configured:
        - offset of outer-sequence to `batch`: [0, 5, 8, 10, 14]
        - offset of inner-sequence to `batch`: [0, 2, 3, 5, 7, 8, 10, 13, 14]
    3. for both `paddle_ivector` and `paddle_matrix`, sequential information needs to be configured **twice**, using the code below, to configure the outer-sequence and inner-sequence, to make it a double-layer sequence input. 

    ```c
    // set the sequence start positions for the outter sequences.
    int outter_seq_pos_array[] = {0, 5, 8, 10, 14};
    paddle_ivector seq_pos =
        paddle_ivector_create(outter_seq_pos_array,
                              sizeof(outter_pos_array) / sizeof(int),
                              false,
                              false);
    // The third parameter of this API indicates the sequence level.
    // 0 for the outter sequence. 1 for the inner sequence.
    // If the input is a sequence not the nested sequence, the third parameter is
    // fixed to be 0.
    CHECK(paddle_arguments_set_sequence_start_pos(in_args, 0, 0, seq_pos));

    // set the sequence start positions for the outter sequences.
    int inner_seq_pos_array[] = {0, 2, 3, 5, 7， 8， 10， 13， 14};
    paddle_ivector seq_pos = paddle_ivector_create(
        inner_pos_array, sizeof(inner_pos_array) / sizeof(int), false, false);
    // The third parameter of this API indicates the sequence level.
    // 0 for the outter sequence. 1 for the inner sequence.
    CHECK(paddle_arguments_set_sequence_start_pos(in_args, 0, 1, seq_pos));
    ```

Notice that:
1. Each `batch` **shall not contain any `0`-length sequence input**. Different layers may have different strategies in dealing with zero sequences. To prevent some unknown errors, please double-check that the input data contains no 0-length sequences. 

### Data-type in Python API
The table below lists the data type in python API (`type` in `paddle.layer.data`), and its corresponding C-API data type:

<html>
<table border="2" frame="border">
<table>
<thead>
<tr>
<th style="text-align:left">Python data type</th>
<th style="text-align:left">C-API data type</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left">paddle.data_type.integer_value</td>
<td style="text-align:left">integer array，without sequence information</td>
</tr>
<tr>
<td style="text-align:left">paddle.data_type.dense_vector</td>
<td style="text-align:left">dense float matrix，without sequence information</td>
</tr>
<tr>
<td style="text-align:left">paddle.data_type.sparse_binary_vector</td>
<td style="text-align:left">sparse float matrix，with binary values (0/1)，without sequence information</td>
</tr>
<tr>
<td style="text-align:left">paddle.data_type.sparse_vector</td>
<td style="text-align:left">sparse float matrix，non-zero values shall be provided，without sequence information</td>
</tr>
<tr>
<td style="text-align:left">paddle.data_type.integer_value_sequence</td>
<td style="text-align:left">integer array，with sequence information</td>
</tr>
<tr>
<td style="text-align:left">paddle.data_type.dense_vector_sequence</td>
<td style="text-align:left">dense float matrix，with sequence information</td>
</tr>
<tr>
<td style="text-align:left">paddle.data_type.sparse_binary_vector_sequence</td>
<td style="text-align:left">sparse float matrix，with binary values (0/1) and sequence information</td>
</tr>
<tr>
<td style="text-align:left">paddle.data_type.sparse_vector_sequence</td>
<td style="text-align:left">sparse float matrix，non-zero values shall be provided，with sequence information</td>
</tr>
<tr>
<td style="text-align:left">paddle.data_type.integer_value_sub_sequence</td>
<td style="text-align:left">integer array，with double-layer sequence information</td>
</tr>
<tr>
<td style="text-align:left">paddle.data_type.dense_vector_sub_sequence</td>
<td style="text-align:left">dense float matrix，with double-layer sequence information</td>
</tr>
<tr>
<td style="text-align:left">paddle.data_type.sparse_binary_vector_sub_sequence</td>
<td style="text-align:left">sparse float matrix，with binary values (0/1)，with double-layer sequence information</td>
</tr>
<tr>
<td style="text-align:left">paddle.data_type.sparse_vector_sub_sequence</td>
<td style="text-align:left">sparse float matrix，non-zero values shall be provided，with double-layer sequence information</td>
</tr>
</tbody>
</table>
</html>
<br>

### Output data
Paddlepaddle has an analogy mechanism in organizing output data to input data. The output data is alos based on `argument` in implement, while `arugment` store output data via `paddle_ivector` and `paddle_matrix`. If the output is a sequence, then `sequence_start_positions` is also provided by layers. Users can directly access to the output information using C-API.

### Summary
- In the inner-implement of PaddlePaddle, both input/output of layers in neural networks are organized as `argument`
- `argument` logically organize input/output information as a unit, and do not really store those data.
- `argument` is based on `paddle_ivector` and `paddle_matrix` to store one-dimensional integer array and two-dimensional float matrix respectively.
- `sequence start positions` contains the sequence inforamtion of input/output data.

We recommend users reading following articles for more details:
1. create `arguemnt` for input/output
    - please read [argument.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/capi/arguments.h)
2. create `paddle_matrix` or `paddle_ivector` for `argument`
    - please read [vector.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/capi/vector.h) for creating `paddle_ivector`
    - please read [matrix.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/capi/matrix.h) for creating `paddle_matrix`
3. create `sequence_start_positions` information for sequential input
    - create `sequence_start_positions` for `argument`: [`paddle_arguments_set_sequence_start_pos`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/capi/arguments.h#L137)
    - read `sequence_start_positions` for `argument`:[`paddle_arguments_get_sequence_start_pos`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/capi/arguments.h#L150)
    - description of argument API: [`paddle_arguments_get_sequence_start_pos`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/capi/arguments.h#L150)
