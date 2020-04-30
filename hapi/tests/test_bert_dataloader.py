import paddle
from hapi.model import set_device
from hapi.text.bert.dataloader import SingleSentenceDataLoader
import hapi.text.tokenizer.tokenization as tokenization

device = set_device("cpu")
paddle.fluid.enable_dygraph(device)

tokenizer = tokenization.FullTokenizer(
    vocab_file="./tmp/hapi/data/pretrained_models/uncased_L-12_H-768_A-12/vocab.txt",
    do_lower_case=True)

bert_dataloader = SingleSentenceDataLoader(
    "./tmp/hapi/aaa.txt",
    tokenizer, ["1", "2"],
    max_seq_length=32,
    batch_size=1)

for data in bert_dataloader.dataloader():
    print(data)
