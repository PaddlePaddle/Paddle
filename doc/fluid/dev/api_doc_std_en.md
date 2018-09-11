# API Doc Standard

- [API Doc Standard](#api-doc-standard)
    - [API Doc Structure](#api-doc-structure)
    - [Format and Examples](#format-and-examples)
    - [Complete Example](#complete-example)


## API Doc Structure

API Doc should contain the following parts(please write them in order):

- Python API Definition

  The definition of API

- Function Description

  Description of API's function. 
  The description includes: meaning, purpose and operation on input of API, reference and corresponding link(if any), formula(if necessary) and explanations of key variables in the formula.

- Args Description

  Description of API parameters.
  Introduce parameters one by one according to the order in API definition.
  The introduction includes: data type, default value(if any), meaning, etc.

- Returns

  Introduction of API returned value.
  Introduce meaning of returned value, provide correspoding format if necessary.
  If returned value is a tuple containing multiple parameters, then introduce parameters one by one in order.

- Raises（if any）

   Abnormality, error that may occur, and possible reasons. If there are more than one possible abnormity or error, they should be listed in order. 

- Note（if any）

  Matters needing attention. If there are more than one matters, they should be listed in order. 

- Examples

  Examples of how to use API.


## Format and Examples

API documentation must obey reStructuredText format, please refer to [here](http://sphinx-doc-zh.readthedocs.io/en/latest/rest.html).
Format and examples of each part of API documantation are as follows: (take fc for example)

- Python API Definition

  - Format

      [Python API Definition]

  - Example

      ```
      fc(input,
         size,
         num_flatten_dims=1,
         param_attr=None,
         bias_attr=None,
         act=None,
         name=None,
         main_program=None,
         startup_program=None)
      ```

- Function Description

  - Format

      This part contains (please write them in order):

      [Function Description]

      [Formula]

      [Symbols' Descriptions if necessary]

      [References if necessary]

  - Example

      [Function Description]

       ```
       **Fully Connected Layer**

       The fully connected layer can take multiple tensors as its inputs. It
       creates a variable called weights for each input tensor, which represents
       a fully connected weight matrix from each input unit to each output unit.
       The fully connected layer multiplies each input tensor with its coresponding
       weight to produce an output Tensor. If multiple input tensors are given,
       the results of multiple multiplications will be sumed up. If bias_attr is
       not None, a bias variable will be created and added to the output. Finally,
       if activation is not None, it will be applied to the output as well.
       ```

      [Formula]

      ```
      This process can be formulated as follows:

      .. math::

           Out = Act({\sum_{i=0}^{N-1}X_iW_i + b})
      ```

      [Symbols' Descriptions if necessary]

      ```
      In the above equation:

      * :math:`N`: Number of the input.
      * :math:`X_i`: The input tensor.
      * :math:`W`: The weights created by this layer.
      * :math:`b`: The bias parameter created by this layer (if needed).
      * :math:`Act`: The activation function.
      * :math:`Out`: The output tensor.
      ```

      [References if necessary]

      Since there is no need for reference of fc, we omit them here. Under other circumstances, please provide explicit reference and link, take layer_norm for example: 

      ```
      Refer to `Layer Normalization <https://arxiv.org/pdf/1607.06450v1.pdf>`_ for more details.
      ```


- Args Description

  - Format

      \[Arg's Name\][(Data Type, Default Value)][Description]

  - Example

      part of fc parameters are as follows:

      ```
      Args:
          input (Variable|list of Variable): The input tensor(s) of this layer, and the dimension of
              the input tensor(s) is at least 2.
          param_attr (ParamAttr|list of ParamAttr, default None): The parameter attribute for learnable
              parameters/weights of this layer.
          name (str, default None): The name of this layer.
      ```

- Returns

  - Format

      [Name][Shape]

  - Example

      ```
      Returns:
          A tensor variable storing the transformation result.
      ```

      when returned value contain more than one tuple, please introduce every parameter in order, take dynamic_lstm for example:

      ```
      Returns:
          A tuple containing:
            The hidden state of LSTM whose shape is (T X D).
            The cell state of LSTM whose shape is (T X D).
      ```

- Raises

  - Format

      [Exception Type][Condition]

  - Example

      ```
      Raises:
          ValueError: If the rank of the input is less than 2.
      ```

- Note

  - Format

     [Note]

  - Example

      there is no Note in fc, so we omit this part. If there is any note, please write clearly. If there are more than one notes, please list them in order. Take scaled\_dot\_product\_attention for example:

      ```
      Note:
          1. When num_heads > 1, three linear projections are learned respectively
             to map input queries, keys and values into queries', keys' and values'.
             queries', keys' and values' have the same shapes with queries, keys
             and values.
          2. When num_heads == 1, scaled_dot_product_attention has no learnable
             parameters.
      ```

- Examples

  - Format

      \[Python Code Snipper]

  - Example

      ```
      Examples:
          .. code-block:: python

            data = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
            fc = fluid.layers.fc(input=data, size=1000, act="tanh")
      ```

## Complete Example

Complete Example of fc please see [here](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/dev/src/fc.py)。
