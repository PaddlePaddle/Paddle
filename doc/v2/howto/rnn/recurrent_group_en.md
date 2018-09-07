# Recurrent Group Tutorial

## Overview

Sequential data is common in natural language processing.

A sentence is a sequence of words and many sentences form a paragraph further. Therefore, a paragraph can be viewed as a nested sequence with two level, where each element of the sequence is another sequence. That is to say, sequential data could be recursive. An example of two-level recursive sequential data is that an article is composed of a sequence of sentences, and each sentence a sequence of words.

PaddlePaddle and PaddlePaddle v2 support two-level recursive sequential data. The two-level sequence is a very flexible data, which helps us to better describe more complex language data such as discribing paragraphs and several rounds of dialogues. Based on two-level sequence input, we can design and build a flexible, hierarchical RNN model that encodes input data from the word and sentence level. For the support of arbitrary levels, please refer to PaddlePaddle Fluid.

In PaddlePaddle, `recurrent_group` is an arbitrarily complex RNN unit. The user only needs to define the calculation that the RNN will complete in one time step. PaddlePaddle is responsible for the propagation of information and error in time series.

Furthermore, `recurrent_group` can also be extended to handle two-level sequence. By defining two nested `recurrent_group` operations at the clause level and the word level respectively, a hierarchical and complex RNN is finally achieved.

Currently, in the PaddlePaddle, there are `recurrent_group` and some Layers that can process bidirectional sequences. For details, refer to the document: <a href = "hierarchical_layer_en.html">Layers for supporting double-layer sequences as input.</a>

## Related Concepts

### Basic Principle 
`recurrent_group` is an arbitrarily complex RNN unit supported by PaddlePaddle. The user only needs to focus on the calculations that the RNN is designed to complete within a single time step. The PaddlePaddle is responsible for completing the propagation of information and gradients over time.

In PaddlePaddle, a simple call to `recurrent_group` is as follows:

``` python 
recurrent_group(step, input, reverse) 
```
- step: A callable function that defines the calculations completed by the RNN unit within a time step
- input: The input must be a single-layer sequence or a double-layer sequence
- reverse: Whether to process the input sequence in reverse order

The core of using `recurrent_group` is to design the logic of the step function. The step function can be freely combined with various layers supported by PaddlePaddle to complete arbitrary arithmetic logic. The input of `recurrent_group` (input) becomes the input of the step function. Since the step function only focuses on the calculation within one time step of RNN, here `recurrent_group` completes the splitting of the original input data for us.

### Input
The input sequence processed by `recurrent_group` is mainly divided into the following three types:

- **Input Data**: When putting a two-level sequence into `recurrent_group`, it will be disassembled into a single-level sequence. When putting a single-level sequence into `recurrent_group`, it will be disassembled into a non-sequence and then passed to the step function. This process is completely transparent to the user. There are two possible types: 1) User input via data_layer; 2) Output from other layers.
		
- **Read-only Memory Input**: `StaticInput` defines a read-only Memory. The input specified by `StaticInput` will not be disassembled by `recurrent_group`, and each time step of the `recurrent_group` loop will always be able to reference all inputs. It may be a non-sequence or a single-layer sequence.
	  
- **Input of Sequence Generation Task**: `GeneratedInput` is only used to specify input data in a sequence generation task.

### Input Example

Sequence generation tasks mostly follow the encoder-decoer architecture. The encoder and decoder can be arbitrary neural network units capable of processing sequences and RNN is the most popular choice.

Given the encoder output and the current word, the decoder predicts the next most likely word each time. In this structure, the decoder accepts two inputs:

- Target sequence to be generated: a input of the decoder and the basis of the decoder loop. `recurrent_group` will disassemble this input type.

- Encoder output, an non-sequencce or single-sequence: a unbounded memory. Each time step in the decoder loop will reference the entire result and should not be disassembled. This type of input must be specified via `StaticInput`. For more discussion on Unbounded Memory, please refer to the paper [Neural Turning Machine](https://arxiv.org/abs/1410.5401).

In a sequence generation task, the decoder RNN always refers to the word vector of the word predicted at the previous moment as the current time input. `GeneratedInput` will automate this process.

### Output
The `step` function must return the output of one or more Layers. The output of this Layer will be the final output of the entire `recurrent_group`. In the output process, `recurrent_group` will concatenate the output of each time step, which is also transparent to the user.

### Memory
Memory can only be defined and used in `recurrent_group`. Memory cannot exist independently and must point to a layer defined by PaddlePaddle. Memory is referenced to get a momentary output from this layer, so memory can be interpreted as a delay operation.

The user can explicitly specify the output of a layer to initialize the memory. When not specified, memory is initialized to 0 by default.

## Sequence-level RNN Introduction

`recurrent_group` helps us to split the input sequence, merge the output, and loop through the sequence of computational logic.

Using this feature, the two nested `recurrent_group` can handle the nested two-level sequences, implementing sequence-level RNN structures at both the word and sentence levels.

- Word-level RNN:  each state corresponds to a word.
- Sequence-level RNN: a sequence-layer RNN consists of multiple word-layer RNNs. Each word-layer RNN (ie, each state of a sequence-layer RNN) has a subsequence.

For convenience of description, the following takes the NLP task as an example. A paragraph containing a subsequence is defined as a two-level sequence, and a sentence containing a word is defined as a single-layer sequence. Then, the zero-level sequence is a word.

## Usage of Sequence-level RNN

### Usage of Training Process
Using `recurrent_group` requires the following conventions:

- **Single-input Single-output**: Both input and output are single layer sequences.
  - If there are multiple inputs, the number of words in different input sequences must be exactly equal.
  - A single-layer sequence is output, and the number of words in the output sequence is the same as the input sequence.
  - memory: define memory to point to a layer in the step function, get a moment output from this layer by referencing memory to form a recurrent connection. The is_seq parameter of memory must be false. If memory is not defined, the operations within each time step are independent.
  - boot_layer: the initial state of memory, set 0 by default. is_seq in memory must be false.
 
- **Double-input Double-output**: Both input and output are two-level sequence.
  - If there are multiple input sequences, the number of subsequence contained in different inputs must be strictly equal, but the number of words in the subsequence may not be equal.
  - output a two-level sequence. The number of subsequence and the number of words are the same as the specified input sequence and the first input is default.
  - memory: defining memory in the step function, pointing to a layer, by referring to the memory to get the output of this layer at a time, forming a recurrent connection. The memory defined in the outer `recurrent_group` step function can record the state of the previous subsequence, either as a single-level sequence (only as read-only memory) or as a word. If memory is not defined, the operations between subsequence are independent.
  - boot_layer: the initial state of memory. It is either a single-level sequence (only as read-only memory) or a vector. The default is not set, that is, the initial state is 0.

- **Double-input Single-output**: not support for now, and output the error with "In hierachical RNN, all out links should be from sequences now".
 
### Usage of Generation Process
Using `beam_search` need follow those conventions: 

- Word-level RNN: generate the next word from a word.
- Sequence-level RNN: the single-layer RNN generated subsequence is concatenated into a new double-layer sequence. Semantically, there is no case where a subsequence generates the next subseq directly.
