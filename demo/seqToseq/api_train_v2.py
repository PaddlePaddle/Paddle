import sys

import paddle.v2 as paddle


def seqToseq_net(source_dict_dim, target_dict_dim, is_generating=False):
    ### Network Architecture
    word_vector_dim = 512  # dimension of word vector
    decoder_size = 512  # dimension of hidden unit in GRU Decoder network
    encoder_size = 512  # dimension of hidden unit in GRU Encoder network

    beam_size = 3
    max_length = 250

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

    if not is_generating:
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
    else:
        # In generation, the decoder predicts a next target word based on
        # the encoded source sequence and the last generated target word.

        # The encoded source sequence (encoder's output) must be specified by
        # StaticInput, which is a read-only memory.
        # Embedding of the last generated word is automatically gotten by
        # GeneratedInputs, which is initialized by a start mark, such as <s>,
        # and must be included in generation.

        trg_embedding = paddle.layer.GeneratedInputV2(
            size=target_dict_dim,
            embedding_name='_target_language_embedding',
            embedding_size=word_vector_dim)
        group_inputs.append(trg_embedding)

        beam_gen = paddle.layer.beam_search(
            name=decoder_group_name,
            step=gru_decoder_with_attention,
            input=group_inputs,
            bos_id=0,
            eos_id=1,
            beam_size=beam_size,
            max_length=max_length)

        return beam_gen


def main():
    paddle.init(use_gpu=False, trainer_count=1)
    is_generating = False

    # source and target dict dim.
    dict_size = 30000
    source_dict_dim = target_dict_dim = dict_size

    # train the network
    if not is_generating:
        cost = seqToseq_net(source_dict_dim, target_dict_dim)
        parameters = paddle.parameters.create(cost)

        # define optimize method and trainer
        optimizer = paddle.optimizer.Adam(
            learning_rate=5e-5,
            regularization=paddle.optimizer.L2Regularization(rate=8e-4))
        trainer = paddle.trainer.SGD(cost=cost,
                                     parameters=parameters,
                                     update_equation=optimizer)
        # define data reader
        wmt14_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.wmt14.train(dict_size), buf_size=8192),
            batch_size=5)

        # define event_handler callback
        def event_handler(event):
            if isinstance(event, paddle.event.EndIteration):
                if event.batch_id % 10 == 0:
                    print "\nPass %d, Batch %d, Cost %f, %s" % (
                        event.pass_id, event.batch_id, event.cost,
                        event.metrics)
                else:
                    sys.stdout.write('.')
                    sys.stdout.flush()

        # start to train
        trainer.train(
            reader=wmt14_reader, event_handler=event_handler, num_passes=2)

    # generate a english sequence to french
    else:
        # use the first 3 samples for generation
        gen_creator = paddle.dataset.wmt14.gen(dict_size)
        gen_data = []
        gen_num = 3
        for item in gen_creator():
            gen_data.append((item[0], ))
            if len(gen_data) == gen_num:
                break

        beam_gen = seqToseq_net(source_dict_dim, target_dict_dim, is_generating)
        # get the pretrained model, whose bleu = 26.92
        parameters = paddle.dataset.wmt14.model()
        # prob is the prediction probabilities, and id is the prediction word. 
        beam_result = paddle.infer(
            output_layer=beam_gen,
            parameters=parameters,
            input=gen_data,
            field=['prob', 'id'])

        # get the dictionary
        src_dict, trg_dict = paddle.dataset.wmt14.get_dict(dict_size)

        # the delimited element of generated sequences is -1,
        # the first element of each generated sequence is the sequence length
        seq_list = []
        seq = []
        for w in beam_result[1]:
            if w != -1:
                seq.append(w)
            else:
                seq_list.append(' '.join([trg_dict.get(w) for w in seq[1:]]))
                seq = []

        prob = beam_result[0]
        beam_size = 3
        for i in xrange(gen_num):
            print "\n*******************************************************\n"
            print "src:", ' '.join(
                [src_dict.get(w) for w in gen_data[i][0]]), "\n"
            for j in xrange(beam_size):
                print "prob = %f:" % (prob[i][j]), seq_list[i * beam_size + j]


if __name__ == '__main__':
    main()
