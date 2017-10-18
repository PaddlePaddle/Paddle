# Sequence Decoder Design
Sequence Decoder is an operator that help generate sequences, 
it shares much logic with dynamic recurrent op.

In text generation tasks, such as machine translation, 
a neural network is trained to rate candidate words given the context,
a decoder will be used to select out good candidates as the next word and extend the prefix with the candidates.

## Beam Search
Beam search is a heuristic search algorithm that explores a graph by expanding the most promising node in a limited set.

It is the core component of Sequence Decoder.

In the original implemention of `RecurrentGradientMachine`, the beam search is a method in RNN,
because the complexity of the algorithm, the implementation is quite trivial and hard to reuse.

Based on the current refactoring work, 
we have many new concept such as `LoDTensor` and `TensorArray` that can better support sequence usages.

## Demo

Let's start from a simple machine translation model, it has an encoder that learns the semantic vector from a sequence,
and a decoder which uses the Sequence Decoder to generate a new sentence.

### Encoder

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

is_generating = False

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

### Decoder
```python
def generate():
    decoder = pd.sequence_decoder()
    with decoder.step():
        decoder_mem = decoder.memory(init=encoder_ctx)  # mark the memory
        # generated_inputs's lod:
        # <batch, instance, candidates>
        target_word = pd.lookup(trg_embedding, decoder.generated_inputs())
        # expand encoder_ctx's batch to fit target_word's lod
        # in the begining, the encoder_ctx should be
        # lod like: [a, b]
        # selected inputs:
        # [[1 2 3] [3 4]]
        # expand to
        # [[a a a] [b b]]
        encoder_ctx_expanded = pd.lod_expand(encoder_ctx, target_word)
        decoder_input = pd.fc(
            act=pd.activation.Linear(),
            input=[target_word, encoder_ctx],
            size=3 * decoder_dim)
        # cur_mem's lod is [[1 1 1] [1 1]], same as the encoder_ctx_expanded
        # gru_out's lod is the same
        gru_out, cur_mem = pd.gru_step(
            decoder_input, mem=decoder_mem, size=decoder_dim)
        decoder.update(mem)  # tell how to update state
        # scores's lod same as the encoder_ctx_expanded
        scores = pd.fc(
            gru_out,
            size=trg_dic_size,
            bias=None,
            act=pd.activation.Softmax())
        # topk_scores, a tensor, [None, k]
        topk_scores, topk_ids = pd.top_k(scores)
        # selected_ids, selected_scores's lod <batch, instance, candidates>
        selected_ids, selected_scores = decoder.beam_search(
            topk_scores, topk_ids)
```

Let's validate the logic with some simple data, assuming that there are 3 sentences to translate

**initial statistics**

```
3 sentences to translate

encoder_ctx:
  lod = [[0, 1, 2, 3]]
  shape = [3, 128]
decoder_mem:
  lod = [[0, 1, 2, 3], [0, 1, 2, 3]]
  shape = [3, 128]
decoder.generated_inputs()
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
  lod = [[0, 2, 5, 6], [0, 2, 5, 6]] 
  shape = [6, 1]
selected_scores:
  lod = [[0, 2, 5, 6], [0, 2, 5, 6]] 
  shape = [6, 1]
```

**first step**
```
encoder_ctx (updated with cur_mem):
  lod = [[0, 1, 2, 3], [0, 1, 2, 3]]
  shape = [3, 128]
decoder_mem:
  lod = [[0, 1, 2, 3], [0, 1, 2, 3]]
  shape = [3, 128]


```
