#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import shutil
import sys
import os


def usage():
    """
    usage information
    """
    print
    print("please use command: ")
    print(
        "python convert_static_to_dygraph.py input_params_dir output_params_dir"
    )
    print


def convert_static_to_dygraph(static_model_path, dygraph_model_path):
    """
    convert paddle static bert model to dygraph model 
    """

    def mkdir(path):
        if not os.path.isdir(path):
            if os.path.split(path)[0]:
                mkdir(os.path.split(path)[0])
        else:
            return
        os.mkdir(path)

    if os.path.exists(dygraph_model_path):
        shutil.rmtree(dygraph_model_path)
    mkdir(dygraph_model_path)

    if not os.path.exists(static_model_path):
        print("paddle static model path doesn't exist.....")
        return -1

    file_list = []
    for root, dirs, files in os.walk(static_model_path):
        file_list.extend(files)

    os.makedirs(os.path.join(dygraph_model_path, "PretrainModelLayer_0"))
    os.makedirs(
        os.path.join(dygraph_model_path,
                     "PretrainModelLayer_0/BertModelLayer_0"))
    os.makedirs(
        os.path.join(dygraph_model_path,
                     "PretrainModelLayer_0/PrePostProcessLayer_0"))
    os.makedirs(
        os.path.join(
            dygraph_model_path,
            "PretrainModelLayer_0/BertModelLayer_0/PrePostProcessLayer_0"))

    #os.chdir(static_model_path)
    #convert embedding file
    embedding_type = ["word", "pos", "sent"]
    for i in range(3):
        src_name = embedding_type[i] + "_embedding"
        trg_name = "Embedding_" + str(i) + "." + src_name
        shutil.copyfile(
            os.path.join(static_model_path, src_name),
            os.path.join(dygraph_model_path,
                         "PretrainModelLayer_0/BertModelLayer_0/" + trg_name))

    #convert pre_encoder file
    shutil.copyfile(
        os.path.join(static_model_path, "pre_encoder_layer_norm_scale"),
        os.path.join(
            dygraph_model_path,
            "PretrainModelLayer_0/BertModelLayer_0/PrePostProcessLayer_0/LayerNorm_0._layer_norm_scale"
        ))
    shutil.copyfile(
        os.path.join(static_model_path, "pre_encoder_layer_norm_bias"),
        os.path.join(
            dygraph_model_path,
            "PretrainModelLayer_0/BertModelLayer_0/PrePostProcessLayer_0/LayerNorm_0._layer_norm_bias"
        ))

    #convert mask lm params file
    shutil.copyfile(
        os.path.join(static_model_path, "mask_lm_out_fc.b_0"),
        os.path.join(dygraph_model_path,
                     "PretrainModelLayer_0/Layer_0.mask_lm_out_fc.b_0"))
    shutil.copyfile(
        os.path.join(static_model_path, "mask_lm_trans_fc.b_0"),
        os.path.join(dygraph_model_path,
                     "PretrainModelLayer_0/FC_0.mask_lm_trans_fc.b_0"))
    shutil.copyfile(
        os.path.join(static_model_path, "mask_lm_trans_fc.w_0"),
        os.path.join(dygraph_model_path,
                     "PretrainModelLayer_0/FC_0.mask_lm_trans_fc.w_0"))
    shutil.copyfile(
        os.path.join(static_model_path, "mask_lm_trans_layer_norm_bias"),
        os.path.join(
            dygraph_model_path,
            "PretrainModelLayer_0/PrePostProcessLayer_0/LayerNorm_0._layer_norm_bias"
        ))
    shutil.copyfile(
        os.path.join(static_model_path, "mask_lm_trans_layer_norm_scale"),
        os.path.join(
            dygraph_model_path,
            "PretrainModelLayer_0/PrePostProcessLayer_0/LayerNorm_0._layer_norm_scale"
        ))
    shutil.copyfile(
        os.path.join(static_model_path, "next_sent_fc.b_0"),
        os.path.join(dygraph_model_path,
                     "PretrainModelLayer_0/FC_1.next_sent_fc.b_0"))
    shutil.copyfile(
        os.path.join(static_model_path, "next_sent_fc.w_0"),
        os.path.join(dygraph_model_path,
                     "PretrainModelLayer_0/FC_1.next_sent_fc.w_0"))
    shutil.copyfile(
        os.path.join(static_model_path, "pooled_fc.b_0"),
        os.path.join(
            dygraph_model_path,
            "PretrainModelLayer_0/BertModelLayer_0/FC_0.pooled_fc.b_0"))
    shutil.copyfile(
        os.path.join(static_model_path, "pooled_fc.w_0"),
        os.path.join(
            dygraph_model_path,
            "PretrainModelLayer_0/BertModelLayer_0/FC_0.pooled_fc.w_0"))

    encoder_num = 0
    for f in file_list:
        if not f.startswith("encoder_layer"):
            continue
        layer_num = f.split('_')[2]
        if int(layer_num) > encoder_num:
            encoder_num = int(layer_num)

    encoder_num += 1
    for i in range(encoder_num):
        encoder_dir = "EncoderSubLayer_" + str(i)
        os.makedirs(
            os.path.join(dygraph_model_path,
                         "PretrainModelLayer_0/BertModelLayer_0/" +
                         "EncoderLayer_0/", encoder_dir))
        os.makedirs(
            os.path.join(dygraph_model_path,
                         "PretrainModelLayer_0/BertModelLayer_0/" +
                         "EncoderLayer_0/", encoder_dir +
                         "/PositionwiseFeedForwardLayer_0"))
        os.makedirs(
            os.path.join(
                dygraph_model_path, "PretrainModelLayer_0/BertModelLayer_0/" +
                "EncoderLayer_0/", encoder_dir + "/MultiHeadAttentionLayer_0"))
        os.makedirs(
            os.path.join(
                dygraph_model_path, "PretrainModelLayer_0/BertModelLayer_0/" +
                "EncoderLayer_0/", encoder_dir + "/PrePostProcessLayer_1"))
        os.makedirs(
            os.path.join(
                dygraph_model_path, "PretrainModelLayer_0/BertModelLayer_0/" +
                "EncoderLayer_0/", encoder_dir + "/PrePostProcessLayer_3"))

    encoder_map_dict = {
        "ffn_fc_0.b_0":
        ("PositionwiseFeedForwardLayer_0", "FC_0.ffn_fc_0.b_0"),
        "ffn_fc_0.w_0":
        ("PositionwiseFeedForwardLayer_0", "FC_0.ffn_fc_0.w_0"),
        "ffn_fc_1.b_0":
        ("PositionwiseFeedForwardLayer_0", "FC_1.ffn_fc_1.b_0"),
        "ffn_fc_1.w_0":
        ("PositionwiseFeedForwardLayer_0", "FC_1.ffn_fc_1.w_0"),
        "multi_head_att_key_fc.b_0":
        ("MultiHeadAttentionLayer_0", "FC_1.key_fc.b_0"),
        "multi_head_att_key_fc.w_0":
        ("MultiHeadAttentionLayer_0", "FC_1.key_fc.w_0"),
        "multi_head_att_output_fc.b_0":
        ("MultiHeadAttentionLayer_0", "FC_3.output_fc.b_0"),
        "multi_head_att_output_fc.w_0":
        ("MultiHeadAttentionLayer_0", "FC_3.output_fc.w_0"),
        "multi_head_att_query_fc.b_0":
        ("MultiHeadAttentionLayer_0", "FC_0.query_fc.b_0"),
        "multi_head_att_query_fc.w_0":
        ("MultiHeadAttentionLayer_0", "FC_0.query_fc.w_0"),
        "multi_head_att_value_fc.b_0":
        ("MultiHeadAttentionLayer_0", "FC_2.value_fc.b_0"),
        "multi_head_att_value_fc.w_0":
        ("MultiHeadAttentionLayer_0", "FC_2.value_fc.w_0"),
        "post_att_layer_norm_bias":
        ("PrePostProcessLayer_1", "LayerNorm_0.post_att_layer_norm_bias"),
        "post_att_layer_norm_scale":
        ("PrePostProcessLayer_1", "LayerNorm_0.post_att_layer_norm_scale"),
        "post_ffn_layer_norm_bias":
        ("PrePostProcessLayer_3", "LayerNorm_0.post_ffn_layer_norm_bias"),
        "post_ffn_layer_norm_scale":
        ("PrePostProcessLayer_3", "LayerNorm_0.post_ffn_layer_norm_scale")
    }

    for f in file_list:
        if not f.startswith("encoder_layer"):
            continue
        layer_num = f.split('_')[2]
        suffix_name = "_".join(f.split('_')[3:])
        in_dir = encoder_map_dict[suffix_name][0]
        rename = encoder_map_dict[suffix_name][1]
        encoder_layer = "EncoderSubLayer_" + layer_num
        shutil.copyfile(
            os.path.join(static_model_path, f),
            os.path.join(
                dygraph_model_path,
                "PretrainModelLayer_0/BertModelLayer_0/EncoderLayer_0/" +
                encoder_layer + "/" + in_dir + "/" + rename))


if __name__ == "__main__":

    if len(sys.argv) < 3:
        usage()
        exit(1)
    static_model_path = sys.argv[1]
    dygraph_model_path = sys.argv[2]
    convert_static_to_dygraph(static_model_path, dygraph_model_path)
