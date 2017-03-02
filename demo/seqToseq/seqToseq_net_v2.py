import paddle.v2.activation as activation
import paddle.v2.attr as attr
import paddle.v2.data_type as data_type
import paddle.v2.layer as layer
import paddle.v2.networks as networks


def seqToseq_net_v2(source_dict_dim, target_dict_dim):
    ### Network Architecture
    word_vector_dim = 512  # dimension of word vector
    decoder_size = 512  # dimension of hidden unit in GRU Decoder network
    encoder_size = 512  # dimension of hidden unit in GRU Encoder network

    #### Encoder
    src_word_id = layer.data(
        name='source_language_word',
        type=data_type.dense_vector(source_dict_dim))
    src_embedding = layer.embedding(
        input=src_word_id,
        size=word_vector_dim,
        param_attr=attr.ParamAttr(name='_source_language_embedding'))
    src_forward = networks.simple_gru(input=src_embedding, size=encoder_size)
    src_backward = networks.simple_gru(
        input=src_embedding, size=encoder_size, reverse=True)
    encoded_vector = layer.concat(input=[src_forward, src_backward])

    #### Decoder
    with layer.mixed(size=decoder_size) as encoded_proj:
        encoded_proj += layer.full_matrix_projection(input=encoded_vector)

    backward_first = layer.first_seq(input=src_backward)

    with layer.mixed(size=decoder_size, act=activation.Tanh()) as decoder_boot:
        decoder_boot += layer.full_matrix_projection(input=backward_first)

    def gru_decoder_with_attention(enc_vec, enc_proj, current_word):

        decoder_mem = layer.memory(
            name='gru_decoder', size=decoder_size, boot_layer=decoder_boot)

        context = networks.simple_attention(
            encoded_sequence=enc_vec,
            encoded_proj=enc_proj,
            decoder_state=decoder_mem)

        with layer.mixed(size=decoder_size * 3) as decoder_inputs:
            decoder_inputs += layer.full_matrix_projection(input=context)
            decoder_inputs += layer.full_matrix_projection(input=current_word)

        gru_step = layer.gru_step(
            name='gru_decoder',
            input=decoder_inputs,
            output_mem=decoder_mem,
            size=decoder_size)

        with layer.mixed(
                size=target_dict_dim, bias_attr=True,
                act=activation.Softmax()) as out:
            out += layer.full_matrix_projection(input=gru_step)
        return out

    decoder_group_name = "decoder_group"
    group_input1 = layer.StaticInputV2(input=encoded_vector, is_seq=True)
    group_input2 = layer.StaticInputV2(input=encoded_proj, is_seq=True)
    group_inputs = [group_input1, group_input2]

    trg_embedding = layer.embedding(
        input=layer.data(
            name='target_language_word',
            type=data_type.dense_vector(target_dict_dim)),
        size=word_vector_dim,
        param_attr=attr.ParamAttr(name='_target_language_embedding'))
    group_inputs.append(trg_embedding)

    # For decoder equipped with attention mechanism, in training,
    # target embeding (the groudtruth) is the data input,
    # while encoded source sequence is accessed to as an unbounded memory.
    # Here, the StaticInput defines a read-only memory
    # for the recurrent_group.
    decoder = layer.recurrent_group(
        name=decoder_group_name,
        step=gru_decoder_with_attention,
        input=group_inputs)

    lbl = layer.data(
        name='target_language_next_word',
        type=data_type.dense_vector(target_dict_dim))
    cost = layer.classification_cost(input=decoder, label=lbl)

    return cost
