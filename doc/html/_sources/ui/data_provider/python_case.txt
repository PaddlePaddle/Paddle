# Python Use Case #

This tutorial guides you into using python script that converts user input data into PaddlePaddle Data Format. 

## Quick Start ##

We use a custom data to show the quick usage. This data consists of two parts with semicolon-delimited `';'`: a) label with 2 dimensions, b) continuous features with 9 dimensions:

    1;0 0 0 0 0.192157 0.070588 0.215686 0.533333 0
    0;0 0 0 0.988235 0.913725 0.329412 0.376471 0 0

The `simple_provider.py` defines a python data provider:

```python
from trainer.PyDataProviderWrapper import DenseSlot, IndexSlot, provider

@provider([DenseSlot(9), IndexSlot(2)])
def process(obj, file_name):
    with open(file_name, 'r') as f:
        for line in f:
        line = line.split(";")
        label = int(line[0])
        image = [float(x) for x in line[1].split()[1:]]
        yield label, image
```

- `@provider`: specify the SlotType and its dimension. Here, we have 2 Slots, DenseSlot(9) stores continuous features with 9 dimensions, and IndexSlot(2) stores label with 2 dimensions. 
- `process`: a generator using **yield** keyword to return results one by one. Here, the return format is 1 Discrete Feature and a list of 9 float Continuous Features.

The corresponding python **Train** data source `define_py_data_source` is:

```python
define_py_data_sources('train.list', None, 'simple_provider', 'process')
```
See <a href = "../trainer_config_helpers_api.html#trainer_config_helpers.data_sources.define_py_data_sources">here</a> for detail API reference of `define_py_data_sources`.

## Sequence Example ##

In some tasks such as Natural Language Processing (NLP), the dimension of Slot is related to the dictionary size, and the dictionary should be dynamically loaded during training or generating. PyDataProviderWrapper can satisfy all these demands easily.

### Sequence has no sub-sequence ###
Following is an example of data provider when using LSTM network to do sentiment analysis (If you want to understand the whole details of this task, please refer to [Sentiment Analysis Tutorial](../demo/sentiment_analysis/index.md)). 

The input data consists of two parts with two-tabs-delimited: a) label with 2 dimensions, b) sequence with dictionary length dimensions: 

    0		I saw this movie at the AFI Dallas festival . It all takes place at a lake house and it looks wonderful .
    1		This documentary makes you travel all around the globe . It contains rare and stunning sequels from the wilderness .
    ...

The `dataprovider.py` in `demo/sentiment` is:

```python
from trainer.PyDataProviderWrapper import *

@init_hook_wrapper
def hook(obj, dictionary, **kwargs):
    obj.word_dict = dictionary
    obj.slots = [IndexSlot(len(obj.word_dict)), IndexSlot(2)]
    obj.logger.info('dict len : %d' % (len(obj.word_dict)))

@provider(use_seq=True, init_hook=hook)
# @provider(use_seq=True, init_hook=hook, pool_size=PoolSize(5000))
def process(obj, file_name):
    with open(file_name, 'r') as fdata:
        for line_count, line in enumerate(fdata):
            label, comment = line.strip().split('\t\t')
            label = int(''.join(label.split(' ')))
            words = comment.split()
            word_slot = [obj.word_dict[w] for w in words if w in obj.word_dict]
            yield word_slot, [label]
```

- `hook`: Initialization hook of data provider. Here, it reads the dictionary, sets the obj.slots based on the dictionary length, and uses obj.logger to output some logs.
- `process`: Here, as the Sequence Mode of input is **Seq** and SlotType is IndexSlot, use_seq is set to True, and the yield format is `[int, int, ....]`.
- `PoolSize`: If there are a lot of data, you may need this argument to increase loading speed and reduce memory footprint. Here, PoolSize(5000) means read at most 5000 samples to memory.

The corresponding python **Train/Test** data sources `define_py_data_sources` is:

```python
train_list = train_list if not is_test else None
word_dict = dict()
with open(dict_file, 'r') as f:
    for i, line in enumerate(open(dict_file, 'r')):
        word_dict[line.split('\t')[0]] = i 

define_py_data_sources(train_list, test_list, module = "dataprovider", obj = "processData",
                       args = {'dictionary': word_dict}, train_async = True)
```

### Sequence has sub-sequence ###

If the sequence of above input data is considered as several sub-sequences joint by dot `'.'`, quesion mark `'?'` or exclamation mark `'!'`, see `processData2` in `demo/sentiment/dataprovider.py` as follows:

```python
import re

@provider(use_seq=True, init_hook=hook)
def process2(obj, file_name):
    with open(file_name, 'r') as fdata:
    pat = re.compile(r'[^.?!]+[.?!]')
    for line_count, line in enumerate(fdata):
        label, comment = line.strip().split('\t\t')
        label = int(''.join(label.split(' ')))
        words_list = pat.findall(comment)
        word_slot_list = [[obj.word_dict[w] for w in words.split() \
                          if w in obj.word_dict] for words in words_list]
        yield word_slot_list, [[label]]
```

- `hook`: the same as above. Note that as **SubSeq Slot must put before Seq Slot** in PaddlePaddle, we could not reverse the yield order in this case. 
- `process2`: Here, as the Sequence Mode of input is **SubSeq**, and the SlotType is IndexSlot, use_seq is set to True, and the yield format is `[[int, int, ...], [int, int, ...], ... ]`.
- `define_py_data_sources`: the same as above.
