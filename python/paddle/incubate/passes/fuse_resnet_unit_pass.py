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


def set_resnet_unit_attrs(resnet_unit, has_shortcut):
    resnet_unit.SetAttr("fuse_add", False)
    resnet_unit.SetAttr("act_type", "relu")
    resnet_unit.SetAttr("has_shortcut", has_shortcut)
    resnet_unit.SetAttr("data_format", 'NHWC')
    resnet_unit.SetAttr("dilation", 1)
    resnet_unit.Attr("stride").MappedPattern(
        op="conv2d", name="strides", element_index=0)
    resnet_unit.Attr("padding").MappedPattern(
        op="conv2d", name="paddings", element_index=0)
    resnet_unit.Attr("group").MappedPattern(op="conv2d", name="groups")
    resnet_unit.Attr("op_device").MappedPattern(op="conv2d", name="op_device")
    resnet_unit.Attr("op_namescope").MappedPattern(
        op="conv2d", name="op_namescope")
    resnet_unit.Attr("momentum").MappedPattern(op="batch_norm", name="momentum")
    resnet_unit.Attr("epsilon").MappedPattern(op="batch_norm", name="epsilon")
    resnet_unit.Attr("use_global_stats").MappedPattern(
        op="batch_norm", name="use_global_stats")


def set_resnet_unit_outputs(resnet_unit, meanX, varX, meanZ=None, varZ=None):
    resnet_unit.SetOutputs(
        RunningMeanX=meanX,
        RunningVarX=varX,
        RunningMeanZ=meanZ,
        RunningVarZ=varZ)


@ir.RegisterPass
def fuse_resnet_unit():
    def pattern_conv_bn(x, filter, scale, bias, mean, var):
        filter.Attr("shape")[0].Mod(32).EQ(0)
        filter.Attr("shape")[1].Mod(8).EQ(0)
        filter.Attr("shape")[2].EQ(1)
        filter.Attr("shape")[3].EQ(1)
        conv2d = ir.PassDesc.OP.conv2d(Input=x, Filter=filter)
        conv2d.SetAttr("data_format", 'NHWC')
        bn = ir.PassDesc.OP.batch_norm(
            X=conv2d, Bias=bias, Mean=mean, Scale=scale, Variance=var)
        return bn

    def pattern_one_input(x, filter, scale, bias, mean, var):
        bn = pattern_conv_bn(x, filter, scale, bias, mean, var)
        relu = ir.PassDesc.OP.relu(X=bn.Output("Y"))
        return relu

    def replace_one_input(x, filter, scale, bias, mean, var):
        resnet_unit = ir.PassDesc.OP.resnet_unit(
            X=x, FilterX=filter, ScaleX=scale, BiasX=bias, MeanX=mean, VarX=var)
        set_resnet_unit_attrs(resnet_unit, False)
        set_resnet_unit_outputs(resnet_unit, mean, var)
        return resnet_unit.Output("Y")

    def pattern_two_input(x, filterX, scaleX, biasX, meanX, varX, z, filterZ,
                          scaleZ, biasZ, meanZ, varZ):
        bnX = pattern_conv_bn(x, filterX, scaleX, biasX, meanX, varX)
        bnZ = pattern_conv_bn(x, filterZ, scaleZ, biasZ, meanZ, varZ)
        ewadd = ir.PassDesc.OP.elementwise_add(
            X=bnX.Output("Y"), Y=bnZ.Output("Y"))
        relu = ir.PassDesc.OP.relu(X=ewadd)
        return relu

    def replace_two_input(x, filterX, scaleX, biasX, meanX, varX, z, filterZ,
                          scaleZ, biasZ, meanZ, varZ):
        resnet_unit = ir.PassDesc.OP.resnet_unit(
            X=x,
            FilterX=filterX,
            ScaleX=scaleX,
            BiasX=biasX,
            MeanX=meanX,
            VarX=varX,
            Z=z,
            FilterZ=filterZ,
            ScaleZ=scaleZ,
            BiasZ=biasZ,
            MeanZ=meanZ,
            VarZ=varZ)
        set_resnet_unit_attrs(resnet_unit, True)
        set_resnet_unit_outputs(resnet_unit, meanX, varX, meanZ, varZ)
        return resnet_unit.Output("Y")

    return (pattern_one_input, replace_one_input), (pattern_two_input,
                                                    replace_two_input)
