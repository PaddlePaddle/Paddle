# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
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

import paddle
import paddle.fluid.ir as ir


@ir.RegisterPass
def fuse_resenet_unit():
    def pattern0(x, filter, scale, bias, mean, var):
        conv2d = ir.PassDesc.OP.conv2d(Input=x, Filter=filter)
        bn = ir.PassDesc.OP.batch_norm(
            X=conv2d, Bias=bias, Mean=mean, Scale=scale, Variance=var)
        relu = ir.PassDesc.OP.relu(X=bn.Output("Y"))
        return relu

    def replace0(x, filter, scale, bias, mean, var):
        resnet_unit = ir.PassDesc.OP.resnet_unit(
            X=x, FilterX=filter, ScaleX=scale, BiasX=bias, MeanX=mean, VarX=var)
        resnet_unit.SetAttr("fuse_add", False)
        resnet_unit.SetAttr("act_type", "relu")
        # resnet_unit.SetAttr("has_shortcut", False)
        # resnet_unit.Attr("stride").MappedPattern(op="conv2d", name="strides", element_index=0)
        # resnet_unit.Attr("padding").MappedPattern(op="conv2d", name="paddings", element_index=0)
        resnet_unit.Attr("group").MappedPattern(op="conv2d", name="groups")
        resnet_unit.Attr("op_device").MappedPattern(
            op="conv2d", name="op_device")
        resnet_unit.Attr("data_format").MappedPattern(
            op="conv2d", name="data_format")
        resnet_unit.Attr("op_namescope").MappedPattern(
            op="conv2d", name="op_namescope")
        resnet_unit.Attr("momentum").MappedPattern(
            op="batch_norm", name="momentum")
        resnet_unit.Attr("epsilon").MappedPattern(
            op="batch_norm", name="epsilon")
        resnet_unit.Attr("use_global_stats").MappedPattern(
            op="batch_norm", name="use_global_stats")
        return resnet_unit.Output("Y")

    def pattern1(x, filter0, scale0, bias0, mean0, var0, y, filter1, scale1,
                 bias1, mean1, var1):
        conv2d_0 = ir.PassDesc.OP.conv2d(Input=x, Filter=filter0)
        bn_0 = ir.PassDesc.OP.batch_norm(
            X=conv2d_0, Bias=bias0, Mean=mean0, Scale=scale0, Variance=var0)
        conv2d_1 = ir.PassDesc.OP.conv2d(Input=y, Filter=filter1)
        bn_1 = ir.PassDesc.OP.batch_norm(
            X=conv2d_1, Bias=bias1, Mean=mean1, Scale=scale1, Variance=var1)
        ewadd = ir.PassDesc.OP.elementwise_add(
            X=bn_0.Output("Y"), Y=bn_1.Output("Y"))
        relu = ir.PassDesc.OP.relu(X=ewadd)
        return relu

    def replace1(x, filter0, scale0, bias0, mean0, var0, y, filter1, scale1,
                 bias1, mean1, var1):
        resnet_unit = ir.PassDesc.OP.resnet_unit(
            X=x,
            FilterX=filter0,
            ScaleX=scale0,
            BiasX=bias0,
            MeanX=mean0,
            VarX=var0,
            Z=y,
            FilterZ=filter1,
            ScaleZ=scale1,
            BiasZ=bias1,
            MeanZ=mean1,
            VarZ=var1)
        resnet_unit.SetAttr("fuse_add", True)
        resnet_unit.SetAttr("act_type", "relu")
        # resnet_unit.SetAttr("has_shortcut", False)
        # resnet_unit.Attr("stride").MappedPattern(op="conv2d", name="strides", element_index=0)
        # resnet_unit.Attr("padding").MappedPattern(op="conv2d", name="paddings", element_index=0)
        resnet_unit.Attr("group").MappedPattern(op="conv2d", name="groups")
        resnet_unit.Attr("op_device").MappedPattern(
            op="conv2d", name="op_device")
        resnet_unit.Attr("data_format").MappedPattern(
            op="conv2d", name="data_format")
        resnet_unit.Attr("op_namescope").MappedPattern(
            op="conv2d", name="op_namescope")
        resnet_unit.Attr("momentum").MappedPattern(
            op="batch_norm", name="momentum")
        resnet_unit.Attr("epsilon").MappedPattern(
            op="batch_norm", name="epsilon")
        resnet_unit.Attr("use_global_stats").MappedPattern(
            op="batch_norm", name="use_global_stats")
        return resnet_unit.Output("Y")

    return (pattern0, replace0), (pattern1, replace1)
