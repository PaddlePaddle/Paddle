...  # the settings and define data provider is omitted.
DICT_DIM = 3000  # dictionary dimension.
word_ids = data_layer('word_ids', size=DICT_DIM)

emb = embedding_layer(
    input=word_ids, size=256, param_attr=ParamAttr(sparse_update=True))
emb_sum = pooling_layer(input=emb, pooling_type=SumPooling())
predict = fc_layer(input=emb_sum, size=DICT_DIM, act=Softmax())
outputs(
    classification_cost(
        input=predict, label=data_layer(
            'label', size=DICT_DIM)))
