from paddle.incubate.nn.functional import expand_modality_expert_id
import paddle

num_expert_per_modality = 4
group_size = 10
modality_offset = 3
is_group_expert = True

expert_id = paddle.to_tensor([[0, 1, 2,], [3, 4, 5]], dtype='int32')

expert_id_out = expand_modality_expert_id(expert_id, 
                                          num_expert_per_modality, 
                                          group_size, 
                                          modality_offset,
                                          is_group_expert)

print(expert_id_out)

