# A LoD-based Sequence Decoder 
In tasks such as machine translation and image to text, 
a **sequence decoder** is necessary to generate sequences.

This documentation describes how to implement the sequence decoder as an operator.

## Beam Search based Decoder
The [beam search algorithm](https://en.wikipedia.org/wiki/Beam_search) is necessary when generating sequences, 
it is a heuristic search algorithm that explores the paths by expanding the most promising node in a limited set.

In the old version of PaddlePaddle, a C++ class `RecurrentGradientMachine` implements the general sequence decoder based on beam search, 
due to the complexity, the implementation relays on a lot of special data structures, 
quite trivial and hard to be customized by users.

There are a lot of heuristic tricks in the sequence generation tasks, 
so the flexibility of sequence decoder is very important to users.

During PaddlePaddle's refactoring work,
some new concept is proposed such as [LoDTensor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/lod_tensor.md) and [TensorArray](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/tensor_array.md) that can better support sequence usage,
and they can help to make the implementation of beam search based sequence decoder **more transparent and modular** .

For example, the RNN sates, candidates IDs and probabilities of beam search can be represented as `LoDTensors`;
the selected candidate's IDs in each time step can be stored in a `TensorArray`, and `Packed` to the sentences translated.

## Necessary to change LoD's absolute offset to relative offsets
The current `LoDTensor` is designed to store levels of variable-length sequences,
it stores several arrays of integers each represents a level.

The integers in each level represents the begin and end (not inclusive) offset of a sequence **in the underlying tensor**, 
let's call this format the **absolute-offset LoD** for clear.

The relative-offset LoD can fast retrieve any sequence but fails to represent empty sequences, for example, a two-level LoD is as follows
```python
[[0, 3, 9]
 [0, 2, 3, 3, 3, 9]]
```
The first level tells that there are two sequences:
- the first's offset is `[0, 3)`
- the second's offset is `[3, 9)`

while on the second level, there are several empty sequences that both begin and end at `3`.
It is impossible to tell how many empty second-level sequences exist in the first-level sequences.

There are many scenarios that relay on empty sequence representation,
such as machine translation or image to text, one instance has no translations or the empty candidate set for a prefix.

So let's introduce another format of LoD, 
it stores **the offsets of the lower level sequences** and is called **relative-offset** LoD.

For example, to represent the same sequences of the above data

```python
[[0, 3, 6]
 [0, 2, 3, 3, 3, 9]]
```

the first level represents that there are two sequences, 
their offsets in the second-level LoD is `[0, 3)` and `[3, 5)`.

The second level is the same with the relative offset example because the lower level is a tensor.
It is easy to find out the second sequence in the first-level LoD has two empty sequences.

The following demos are based on relative-offset LoD.

## Usage in a simple machine translation
Let's start from a simple machine translation model that is simplified from [machine translation chapter](https://github.com/PaddlePaddle/book/tree/develop/08.machine_translation) to draw a simple blueprint of what a sequence decoder can do and how to use it.

The model has an encoder that learns the semantic vector from a sequence,
and a decoder which uses the Sequence Decoder to generate new sentences.

**Encoder**

```python
import paddle as pd

dict_size = 8000
source_dict_size = dict_size
target_dict_size = dict_size
word_vector_dim = 128
encoder_dim = 128
decoder_dim = 128
beam_size = 5
max_length = 120

# encoder
src_word_id = pd.data(
    name='source_language_word',
    type=pd.data.integer_value_sequence(source_dict_dim))
src_embedding = pd.embedding(size=source_dict_size, size=word_vector_dim)

src_word_vec = pd.lookup(src_embedding, src_word_id)

encoder_out_seq = pd.gru(input=src_word_vec, size=encoder_dim)

encoder_ctx = pd.last_seq(encoder_out_seq)
# encoder_ctx_proj is the learned semantic vector
encoder_ctx_proj = pd.fc(
    encoder_ctx, size=decoder_dim, act=pd.activation.Tanh(), bias=None)
```

**Decoder**
```python
def generate():
    decoder = pd.sequence_decoder()
    with decoder.step():
        # states for prefixes
        decoder_mem = decoder.memory(init=encoder_ctx)  # mark the memory
        target_word = pd.lookup(trg_embedding, decoder.gendrated_ids())
        # expand encoder_ctx's batch to fit target_word's lod
        # for example
        # decoder_mem.lod is
        # [[0 1 3],
        #  [0 1 3 6]]
        # its tensor content is [a1 a2 a3 a4 a5]
        # which means there are 2 sentences to translate
        #   - the first sentence has 1 translation prefixes, the offsets are [0, 1)
        #   - the second sentence has 2 translation prefixes, the offsets are [1, 3) and [3, 6)
        # the target_word.lod is 
        # [[0, 1, 6]
        #  [0, 2, 4, 7, 9 12]]
        # which means 2 sentences to translate, each has 1 and 5 prefixes
        # the first prefix has 2 candidates
        # the following has 2, 3, 2, 3 candidates
        # the encoder_ctx_expanded's content will be
        # [a1 a1 a2 a2 a3 a3 a3 a4 a4 a5 a5 a5]
        encoder_ctx_expanded = pd.lod_expand(encoder_ctx, target_word)
        decoder_input = pd.fc(
            act=pd.activation.Linear(),
            input=[target_word, encoder_ctx],
            size=3 * decoder_dim)
        gru_out, cur_mem = pd.gru_step(
            decoder_input, mem=decoder_mem, size=decoder_dim)
        decoder.update(mem)  # tell how to update state
        # scores's lod same with the encoder_ctx_expanded
        scores = pd.fc(
            gru_out,
            size=trg_dic_size,
            bias=None,
            act=pd.activation.Softmax())
        # topk_scores, a tensor, [None, k]
        topk_scores, topk_ids = pd.top_k(scores)
        # selected_ids is the selected candidates that will be append to the translation
        # selected_scores is the scores of the selected candidates
        # generated_scores is the score of the translations(with candidates appended)
        selected_ids, selected_scores, generated_scores = decoder.beam_search(
            topk_scores, topk_ids, decoder.generated_scores())
        # the latest value of trans_scores will be cached in decoder.generated_scores()
        # the latest value of selected_ids will be cached in decoder.generated_ids()

        decoder.output(selected_ids)
        decoder.output(selected_scores)
        decoder.output(generated_scores)

translation_word_ids, word_scores, trans_scores = decoder()
```

The implementation of sequence decoder can reuse the C++ class [RNNAlgorithm](https://github.com/Superjom/Paddle/blob/68cac3c0f8451fe62a4cdf156747d6dc0ee000b3/paddle/operators/dynamic_recurrent_op.h#L30),
so the python syntax is quite similar to a [RNN](https://github.com/Superjom/Paddle/blob/68cac3c0f8451fe62a4cdf156747d6dc0ee000b3/doc/design/block.md#blocks-with-for-and-rnnop).

Compared to a RNN, sequence decoder has two special members (that exposed as variables):

1. `decoder.generated_scores()` store the latest scores for candidates set.
2. `decoder.generated_ids()` store the latest candidate word ids.

Both of them are two-level `LoDTensors`

- the first level represents `batch_size` of (source) sentences;
- the second level represents the candidate word ID sets for translation prefix.

for example, 3 source sentences to translate, and has 2, 3, 1 candidates.

Unlike an RNN, in sequence decoder, the previous state and the current state have different LoD and shape,
a `lod_expand` operator is used to expand the LoD of the previous state to fit the current state.

For example, the previous state

* LoD is `[0, 1, 3][0, 2, 5, 6]`
* content of tensor is `a1 a2 b1 b2 b3 c1`

the current state stored in `encoder_ctx_expanded`

* LoD is `[0, 2, 7][0 3 5 8 9 11 11]`
* the content is 
  - a1 a1 a1 (a1 has 3 candidates, so the state should be copied 3 times for each candidates)
  - a2 a2
  - b1 b1 b1
  - b2
  - b3 b3
  - None (c1 has 0 candidates, so c1 is dropped)

Benefit from the relative offset LoD, empty candidate set can be represented naturally.

the status in each time step can be stored in `TensorArray`, and `Pack`ed to a final LoDTensor, the corresponding syntax is 

```python
decoder.output(selected_ids)
decoder.output(selected_scores)
decoder.output(generated_scores)
```

the `selected_ids` is the candidate ids for the prefixes, 
it will be `Packed` by `TensorArray` to a two-level `LoDTensor`,
the first level represents the source sequences,
the second level represents generated sequences.


## Appendix
Let's validate the logic with some simple data, assuming that there are 3 sentences to translate

**initial statistics**

```python
3 sentences to translate

encoder_ctx:
  lod = [[0, 1, 2, 3]]
  shape = [3, 128]
decoder_mem:
  lod = [[0, 1, 2, 3], [0, 1, 2, 3]]
  shape = [3, 128]
decoder.gendrated_ids()
  lod = [[0, 1, 2, 3], [0, 1, 2, 3]]
  shape = [3, 1]
  content = [0, 0, 0] # 0 is the word id of <s>
target_word:
  lod = [[0, 1, 2, 3], [0, 1, 2, 3]]
  shape = [3, 128]
encoder_ctx_expand:
  lod = [[0, 1, 2, 3], [0, 1, 2, 3]]
  shape = [3, 128]
decoder_input:
  lod = [[0, 1, 2, 3], [0, 1, 2, 3]]
  shape = [3, 768]
gru_out:
  lod = [[0, 1, 2, 3], [0, 1, 2, 3]]
  shape = [3, 128]
cur_mem:
  lod = [[0, 1, 2, 3], [0, 1, 2, 3]]
  shape = [3, 128]
scores:
  lod = [[0, 1, 2, 3], [0, 1, 2, 3]]
  shape = [3, 8000]
topk_scores:
  lod = [[0, 1, 2, 3], [0, 1, 2, 3]]
  shape = [3, 5]
topk_ids:
  lod = [[0, 1, 2, 3], [0, 1, 2, 3]]
  shape = [3, 5]
# first instance get 2 candidates
# second instance get 3 candidates
# third instance get 1 candidates
selected_ids:
  lod = [[0, 2, 5, 6], [0, 1, 2, 3, 4, 5, 6]] 
  shape = [6, 1]
selected_scores:
  lod = [[0, 2, 5, 6], [0, 1, 2, 3, 4, 5, 6]] 
  shape = [6, 1]
```

**first step**
```python
encoder_ctx (updated with cur_mem):
  lod = [[0, 1, 2, 3], [0, 1, 2, 3]]
  shape = [3, 128]
decoder_mem:
  lod = [[0, 1, 2, 3], [0, 1, 2, 3]]
  shape = [3, 128]
decoder.gendrated_ids()
  lod = [[0, 2, 5, 6], [0, 1, 2, 3, 4, 5, 6]] 
  shape = [6, 1]
target_words:
  lod = [[0, 2, 5, 6], [0, 1, 2, 3, 4, 5, 6]] 
  shape = [6, 128]
# in the second level of LoD
# the first sequence repeat 2 times
# the second sequence repeat 3 times
# third sequence repeat 1 time
encoder_ctx_expand:
  lod = [[0, 2, 5, 6], [0, 1, 2, 3, 4, 5, 6]] 
  shape = [6, 128]
decoder_input:
  lod = [[0, 2, 5, 6], [0, 1, 2, 3, 4, 5, 6]] 
  shape = [6, 768]
gru_out:
  lod = [[0, 2, 5, 6], [0, 1, 2, 3, 4, 5, 6]] 
  shape = [6, 128]
cur_mem:
  lod = [[0, 2, 5, 6], [0, 1, 2, 3, 4, 5, 6]] 
  shape = [6, 128]
scores:
  lod = [[0, 2, 5, 6], [0, 1, 2, 3, 4, 5, 6]] 
  shape = [6, 8000]
topk_scores:
  lod = [[0, 2, 5, 6], [0, 1, 2, 3, 4, 5, 6]] 
  shape = [6, 5]
topk_ids:
  lod = [[0, 2, 5, 6], [0, 1, 2, 3, 4, 5, 6]] 
  shape = [6, 5]
# there are 6 instances (prefixes) now
# 1-th instance get 5 candidates
# 2-th instance get 2 candidates
# 3-th instance get 4 candidates
# 4-th instance get 2 candidates
# 5-th instance get 4 candidates
# 6-th instance get 1 candidates (may be reatch the end of sequence)
# NOTE each sentence to translate share a beam_size throughout all the instances
selected_ids:
  lod = [[0, 7, 17, 18], # 3 sentence
         [0, 5, 7, 11, 13, 17, 18]] # candidates for each instance
  shape = [18, 1]
selected_scores:
  lod = selected_ids.lod
  shape = selected_ids.shape
```
**second step**
```python
encoder_ctx:
  lod  = [[0, 2, 5, 6], [0, 1, 2, 3, 4, 5, 6]] # equals last cur_mem.lod
  shape = [6, 128]
decoder_mem:
  lod = cur_mem.lod # last step
  shape = cur_mem.shape
decoder.gendrated_ids() # same with last selected_ids
  lod = [[0, 2, 5, 6], # 3 sentence
         [0, 5, 7, 11, 13, 17, 18]] # candidates for each instance
  shape = [18, 1]
target_words:
  lod = decoder.gendrated_ids().lod
  shape = [18, 128]
encoder_ctx_expand:
  # there are 6 instances in the previous encoder_ctx 
  # the 1-th instance repeat 5 times
  # the 2-th repeat 2 times
  # the 3-th repeat 4 times
  # ...
  lod = [[0, 2, 5, 6], # 3 sentence
         [0, 5, 7, 11, 13, 17, 18]] # candidates for each instance
  shape = [18, 128]
decoder_input:
  lod = decoder.gendrated_ids().lod
  shape = [18, 768]
gru_out:
  lod = decoder.gendrated_ids().lod
  shape = [18, 128]
cur_mem:
  lod = decoder.gendrated_ids().lod
  shape = [18, 128]
scores:
  lod = decoder.gendrated_ids().lod
  shape = [18, 8000]
topk_scores:
  lod = decoder.gendrated_ids().lod
  shape = [18, 5]
topk_ids:
  lod = decoder.gendrated_ids().lod
  shape = [18, 5]
# there are 18 instances now
# something special happens
# some instances has no candidates(pruned or reach the end of a sentence)
# the number of candidates are 
# 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2
# LoD can cover these sceneraios naturally.
selected_ids:
  lod = [[0, 6, 9, 18], [0, 0, 1, 3, 3, 4, 6, 6, 7, 9, 9, 10, 12, 12, 13, 15, 15, 16, 18]]
  shape = [18, 1]
selected_scores:
  lod = selected_ids.lod
  shape = selected_ids.shape
```

In conclusion, there are two groups of LoD
1. same with `decoder.gendrated_ids()`
  - `target_words`
  - `encoder_ctx_expand`
  - `decoder_input`
  - `gru_out`
  - `cur_mem`
  - `scores`
  - `topk_scores`
  - `topk_ids`
2. same with `selected_ids` 
  - `selected_scores`
  - all the other tensors in the next time step will be updated to this LoD

The `decoder.output(selected_ids)` will concatenate the `selected_ids` in every step and output as a variable which has the LoD like

```
[[0, 4, 9, 12],
[0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11]]
```

because there are three sentences need to translate, 
so the first level of LoD has four numbers,

- the first sentence has four translation prefixes
  - each has a1, a2-a1, a3-a2, a4-a3 candidate words.
- the second sentence has five translation prefixes
  - each has a5-a4, a6-a5, a7-a6, a8-a7 candidate words.
- the third sentence has three translation prefixes
  - each has a9-a8, a10-a9, a11-a10 candidate words
