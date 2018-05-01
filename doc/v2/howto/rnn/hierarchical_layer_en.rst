Layers supporting hierarchical sequence as input
================================================

..	contents::

summary
=======

The sequence is common data structure in the proccess of NLP(nature language proccess). A single word can be seem as a non-sequence input or a zero-layer input. The sentence make up by words is a one-layer sequence and the senten is a two-layer squence. 

Two-laryer sequence is a recursive sequence, each elecment of it is a single-laryer. It's a very flexible way to combine datas and can help our process some complex input information.

+ zero-layer sequence: a single atom, It can be any input data type support by PaddlePaddle.
+ one-layer sequence: many elements  make up one column,every element is a zero-layer sequence, the order between the elements is the most important information.
+ two-layer sequence: many elements make up one column, every element is a one-layer sequence ,it's called subseq, each element of subseq is a zero-laryer sequence.

In PaddlePaddle the following can accept two-layer as input, and accomplish the correspond calculation.

pooling
=======

The following content is the example of pooling:

..	code-block:: bash

        seq_pool = pooling(input=layer,
                           pooling_type=pooling.Max(),
                           agg_level=AggregateLevel.TO_SEQUENCE)
                           
- `pooling_type` currently supports two types: pooling.Max() and pooling.Avg().

- When ʻagg_level=AggregateLevel.TO_NO_SEQUENCE` (default):

   - Effect: Two-level sequence is converted into a 0-layer sequence after operation, or one-layer sequence is converted into a 0-layer sequence after operation
   - Input: a double-layer sequence, or a single-layer sequence
   - Output: A 0-layer sequence, which is the average (or maximum) of the entire input sequence (single or double)

- When ʻagg_level=AggregateLevel.TO_SEQUENCE` is:

   - Effect: A two-layer sequence is transformed into a one-layer sequence
   - Input: must be a two-layer sequence
   - Output: a one-layer sequence where each element of the sequence is the average (or maximum) value of each subseq element of the original two-layer sequence.
   

last_seq and first_seq
=====================

An example of using last_seq is as follows (first_seq is similar).

.. code-block:: bash

         Last = last_seq(input=layer,
                         Agg_level=AggregateLevel.TO_SEQUENCE)
        
- When ʻagg_level=AggregateLevel.TO_NO_SEQUENCE` (default):

   - Effect: A double-level sequence is converted into a 0-level sequence after operation, or a single-level sequence is converted into a 0-level sequence after operation
   - Input: A double-layered sequence or a single-layered sequence
   - Output: A 0-level sequence, the last, or the first element of the entire input sequence (double or single layer).

- When ʻagg_level=AggregateLevel.TO_SEQUENCE` is:
   - Effect: A two-level sequence is transformed into a one-layer sequence
   - Input: must be a two-layered sequence
   - Output: A single-layer sequence in which each element is the last (or first) element of each subseq in a two-layer sequence.


expand
======

The use of expand is as follows.

.. code-block:: bash

        Ex = expand(input=layer1,
                    Expand_as=layer2,
                    Expand_level=ExpandLevel.FROM_NO_SEQUENCE)
        
- When `expand_level=ExpandLevel.FROM_NO_SEQUENCE` (default):

  - Role: A 0-layer sequence is extended by operation into a one-layer sequence, or a two-layer sequence
  - Input: layer1 must be a 0-level sequence that is the data to be extended; layer2 can be a one-layer sequence, or a two-layer sequence, providing extended length information
  - Output: A one-layer sequence or a two-layer sequence, the type of the output sequence (two-layer sequence or one-layer sequence) and the number of elements contained in the sequence are the same as layer2. If the output is a one-layer sequence, each element of a one-layer sequence (0-layer sequence) is a copy of the layer1 element; if the output is a two-layer sequence, each element in the two-sequence of the two-layer sequence (0-layer sequence) Is a copy of the layer1 element

- When `expand_level=ExpandLevel.FROM_SEQUENCE` is:

  - Role: A one-layer sequence is extended to a two-layer sequence
  - Input: layer1 must be a one-layer sequence that is the data to be extended; layer2 must be a two-level sequence, providing extended length information
  - Output: A two-layer sequence with the same number of elements in the sequence as layer2. It is required that the number of elements contained in a one-layer sequence (0-layer sequence) be the same as the number of bisequences containing subseq. The i-th element of the single-layer sequence (the 0-layer sequence) is expanded into a single-layer sequence that constitutes the i-th subseq of the output two-layer sequence.




