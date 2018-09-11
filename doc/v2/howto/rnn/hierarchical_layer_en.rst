###########################
Layers that Support Hierarchical Sequences as Input
###########################
 
.. contents::
 
Overview 
====
 
A sequence is a common data type in natural language processing tasks. An independent word can be regarded as a non-sequential input or a 0-level sequence. A sentence made up of words is a single-level sequence; a number of sentences make up a paragraph, which is a double-level sequence.
 
A double-level sequence is a nested sequence where each element is a single-level sequence. This is a very flexible way of organizing data that helps us construct some complex input information.
 
We can define non-sequences, single-level sequences, and double-level sequences at the following levels.
 
+ 0-level sequence: an independent element. Its type can be any input data type supported by PaddlePaddle;
+ Single-level sequence: multiple elements arranged in a row; each element is a 0-level sequence. The order of elements is an important input information;
+ Double-level sequence: multiple elements arranged in a row; each element is a single-layer sequence called a subseq of a double-level sequence, and each element of the subseq is a 0-level sequence.
 
In PaddlePaddle, the following layers accept double-layer sequences as input and perform corresponding calculations.
 
`pooling`
========
 
The use of pooling is as follows:
 
.. code-block:: bash
 
        Seq_pool = pooling(input=layer,
                           Pooling_type=pooling.Max(),
                           Agg_level=AggregateLevel.TO_SEQUENCE)
        
- `pooling_type` currently supports two types: pooling.Max() and pooling.Avg().
 
- When ʻagg_level=AggregateLevel.TO_NO_SEQUENCE` (default):
 
  - Effect: a double-level sequence input will be converted into a 0-level sequence, and a single-level sequence will be converted into a 0-level sequence 
  - Input: a double-level sequence or a single-level sequence
  - Output: a 0-level sequence which is the average (or maximum) of the entire input sequence (single or double)
 
- When ʻagg_level=AggregateLevel.TO_SEQUENCE`:
 
  - Effect: a double-level sequence will be transformed into a single-level sequence
  - Input: a double-level sequence
  - Output: a single-level sequence where each element of the sequence is the average (or maximum) value of each subseq element of the original double-level sequence.
 
`last_seq` and `first_seq`
=====================
 
An example of using `last_seq` is as follows (usage of `first_seq` is similar).
 
.. code-block:: bash
 
        Last = last_seq(input=layer,
                        Agg_level=AggregateLevel.TO_SEQUENCE)
        
- When ʻagg_level=AggregateLevel.TO_NO_SEQUENCE` (default):
 
  - Effect: a double-level sequence input will be converted into a 0-level sequence, and a single-level sequence will be converted into a 0-level sequence
  - Input: a double-level sequence or a single-level sequence
  - Output: a 0-level sequence, which is the last or the first element of the input sequence (double or single level).
 
- When ʻagg_level=AggregateLevel.TO_SEQUENCE`:
  - Effect: a double-level sequence will be transformed into a single-level sequence
  - Input: a double-level sequence
  - Output: a single-layer sequence in which each element is the last (or first) element of each subseq in a double-level sequence.
 
`expand`
======
 
The use of expand is as follows.
 
.. code-block:: bash
 
        Ex = expand(input=layer1,
                    Expand_as=layer2,
                    Expand_level=ExpandLevel.FROM_NO_SEQUENCE)
        
- When `expand_level=ExpandLevel.FROM_NO_SEQUENCE` (default):
 
  - Effect: a 0-level sequence is extended to a single-level sequence or a double-level sequence
  - Input: layer1 must be a 0-level sequence to be extended; layer2 can be a single-level sequence or a double-level sequence that provides the extended length information
  - Output: a single-level sequence or a double-level sequence; the type of the output sequence and the number of elements contained in the sequence are the same as layer2. If the output is a single-level sequence, each element of the single-level sequence will be a copy of the layer1 element. If the output is a double-level sequence, each element in the double-level sequence will be a copy of the layer1 element
 
- When `expand_level=ExpandLevel.FROM_SEQUENCE`:
 
  - Effect: a single-level sequence is extended to a double-level sequence
  - Input: layer1 must be a single-level sequence to be extended; layer2 must be a double-level sequence providing extended length information
  - Output: a double-level sequence with the same number of elements as that of layer2. It is required that the number of elements in the single-level sequence be the same as the number of subseq in the double-level sequences. The i-th element of the single-level sequence (the 0-level sequence) is expanded into a single-level sequence that constitutes the i-th subseq of the output, the double-level sequence.
