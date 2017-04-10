import paddle.v2 as paddle
from paddle.v2.config_base import Layer
from paddle.trainer_config_helpers.default_decorators import wrap_name_default
from paddle.trainer_config_helpers.layers import RecurrentLayerGroupSetGenerator, Generator


class BaseGeneratedInputV2(object):
    def __init__(self):
        self.bos_id = None
        self.eos_id = None

    def before_real_step(self):
        raise NotImplementedError()

    def after_real_step(self, *args):
        raise NotImplementedError()


class GeneratedInputV2(BaseGeneratedInputV2):
    def __init__(self, size, embedding_name, embedding_size):
        super(GeneratedInputV2, self).__init__()
        self.size = size
        self.embedding_name = embedding_name
        self.embedding_size = embedding_size

    def after_real_step(self, input):
        return paddle.layer.max_id(input=input, name='__beam_search_predict__')

    def before_real_step(self):
        predict_id = paddle.layer.memory(
            name='__beam_search_predict__',
            size=self.size,
            boot_with_const_id=self.bos_id)

        trg_emb = paddle.layer.embedding(
            input=predict_id,
            size=self.embedding_size,
            param_attr=paddle.attr.ParamAttr(name=self.embedding_name))
        return trg_emb


class RecurrentLayerGroupSetGeneratorV2(Layer):
    def __init__(self, eos_name, max_length, beam_size, num_results_per_sample):
        self.eos_name = eos_name
        self.max_length = max_length
        self.beam_size = beam_size
        self.num_results_per_sample = num_results_per_sample
        super(RecurrentLayerGroupSetGeneratorV2, self).__init__(
            name=eos_name, parent_layers={})

    def to_proto_impl(self, context=None, **kwargs):
        RecurrentLayerGroupSetGenerator(
            Generator(
                eos_layer_name=self.eos_name,
                max_num_frames=self.max_length,
                beam_size=self.beam_size,
                num_results_per_sample=self.num_results_per_sample))
        return self

    def context_name(self):
        return self.eos_name + ".fake"

    def use_context_name(self):
        return True

@wrap_name_default()
def beam_search(step,
                input,
                bos_id,
                eos_id,
                beam_size,
                max_length=500,
                name=None,
                num_results_per_sample=None):
    if num_results_per_sample is None:
        num_results_per_sample = beam_size
    assert num_results_per_sample <= beam_size
        # logger.warning("num_results_per_sample should be less than beam_size")

    if isinstance(input, paddle.layer.StaticInputV2) or isinstance(input, BaseGeneratedInputV2):
        input = [input]

    generated_input_index = -1

    real_input = []
    for i, each_input in enumerate(input):
        assert isinstance(each_input, paddle.layer.StaticInputV2) or isinstance(
            each_input, BaseGeneratedInputV2)
        if isinstance(each_input, BaseGeneratedInputV2):
            assert generated_input_index == -1
            generated_input_index = i
        else:
            real_input.append(each_input)

    assert generated_input_index != -1

    gipt = input[generated_input_index]
    assert isinstance(gipt, BaseGeneratedInputV2)

    gipt.bos_id = bos_id
    gipt.eos_id = eos_id

    def __real_step__(*args):
        eos_name = "__%s_eos_layer__" % name
        generator = RecurrentLayerGroupSetGeneratorV2(
            eos_name, max_length, beam_size, num_results_per_sample)

        args = list(args)
        before_step_layer = gipt.before_real_step()
        before_step_layer.append_child(layer=generator,
                                       parent_names=[before_step_layer.name])
        args.insert(generated_input_index, before_step_layer)

        predict = gipt.after_real_step(step(*args))

        eos = paddle.layer.eos(input=predict, eos_id=eos_id, name=eos_name)
        predict.append_child(layer=eos, parent_names=[predict.name])

        return predict

    # tmp = paddle.layer.recurrent_group(
    #     step=__real_step__,
    #     input=real_input,
    #     reverse=False,
    #     name=name,
    #     is_generating=True)
    tmp = paddle.layer.recurrent_group(
        step=__real_step__,
        input=real_input,
        name=name)

    return tmp
