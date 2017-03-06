import paddle.v2 as paddle


def seqToseq_net_v2(source_dict_dim, target_dict_dim):
    ### Network Architecture
    word_vector_dim = 512  # dimension of word vector
    decoder_size = 512  # dimension of hidden unit in GRU Decoder network
    encoder_size = 512  # dimension of hidden unit in GRU Encoder network

    #### Encoder
    src_word_id = paddle.layer.data(
        name='source_language_word',
        type=paddle.data_type.integer_value_sequence(source_dict_dim))
    src_embedding = paddle.layer.embedding(
        input=src_word_id,
        size=word_vector_dim,
        param_attr=paddle.attr.ParamAttr(name='_source_language_embedding'))
    src_forward = paddle.networks.simple_gru(
        input=src_embedding, size=encoder_size)
    src_backward = paddle.networks.simple_gru(
        input=src_embedding, size=encoder_size, reverse=True)
    encoded_vector = paddle.layer.concat(input=[src_forward, src_backward])

    #### Decoder
    with paddle.layer.mixed(size=decoder_size) as encoded_proj:
        encoded_proj += paddle.layer.full_matrix_projection(
            input=encoded_vector)

    backward_first = paddle.layer.first_seq(input=src_backward)

    with paddle.layer.mixed(
            size=decoder_size, act=paddle.activation.Tanh()) as decoder_boot:
        decoder_boot += paddle.layer.full_matrix_projection(
            input=backward_first)

    def gru_decoder_with_attention(enc_vec, enc_proj, current_word):

        decoder_mem = paddle.layer.memory(
            name='gru_decoder', size=decoder_size, boot_layer=decoder_boot)

        context = paddle.networks.simple_attention(
            encoded_sequence=enc_vec,
            encoded_proj=enc_proj,
            decoder_state=decoder_mem)

        with paddle.layer.mixed(size=decoder_size * 3) as decoder_inputs:
            decoder_inputs += paddle.layer.full_matrix_projection(input=context)
            decoder_inputs += paddle.layer.full_matrix_projection(
                input=current_word)

        gru_step = paddle.layer.gru_step(
            name='gru_decoder',
            input=decoder_inputs,
            output_mem=decoder_mem,
            size=decoder_size)

        with paddle.layer.mixed(
                size=target_dict_dim,
                bias_attr=True,
                act=paddle.activation.Softmax()) as out:
            out += paddle.layer.full_matrix_projection(input=gru_step)
        return out

    decoder_group_name = "decoder_group"
    group_input1 = paddle.layer.StaticInputV2(input=encoded_vector, is_seq=True)
    group_input2 = paddle.layer.StaticInputV2(input=encoded_proj, is_seq=True)
    group_inputs = [group_input1, group_input2]

    trg_embedding = paddle.layer.embedding(
        input=paddle.layer.data(
            name='target_language_word',
            type=paddle.data_type.integer_value_sequence(target_dict_dim)),
        size=word_vector_dim,
        param_attr=paddle.attr.ParamAttr(name='_target_language_embedding'))
    group_inputs.append(trg_embedding)

    # For decoder equipped with attention mechanism, in training,
    # target embeding (the groudtruth) is the data input,
    # while encoded source sequence is accessed to as an unbounded memory.
    # Here, the StaticInput defines a read-only memory
    # for the recurrent_group.
    decoder = paddle.layer.recurrent_group(
        name=decoder_group_name,
        step=gru_decoder_with_attention,
        input=group_inputs)

    lbl = paddle.layer.data(
        name='target_language_next_word',
        type=paddle.data_type.integer_value_sequence(target_dict_dim))
    cost = paddle.layer.classification_cost(input=decoder, label=lbl)

    return cost
