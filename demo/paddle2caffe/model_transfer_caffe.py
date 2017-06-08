import os
import struct
import numpy as np
import caffe

class ModelConfig:
    def __init__(self, path, params):
        self.model_path = path 
        self.model_params = params

def load_model_parameters(conf, layers, output_path):
    net = caffe.Net(conf.model_path, 1)
    if len(layers) == 0: layers = net.params.keys()
    param_num = 0
    for layer_name in layers:
        params = net.params[layer_name]
        layer_name_new = layer_name.replace('/', '-')
        print layer_name
        print layer_name_new
        dict = {"a": "1", "b" : "2", "c" : "3", "d" : "4", "e" : "5", "f" : "6"}
        if "res" in layer_name_new:
            str1, str2 = layer_name_new.split("_") 
            layer_name_new = str1[0:4] + "_" + dict[str1[4]] + "_" + str2 + "_conv" 
        elif "bn" in layer_name_new and "branch" in layer_name_new: 
            str1, str2 = layer_name_new.split("_") 
            layer_name_new = "res" + str1[2] + "_" + dict[str1[3]] + "_" + str2 + "_bn"
        elif "scale" in layer_name_new and "branch" in layer_name_new: 
            str1, str2 = layer_name_new.split("_") 
            layer_name_new = "res" + str1[5] + "_" + dict[str1[6]] + "_" + str2 + "_bn"
        elif layer_name_new == "conv1": 
             layer_name_new = "conv1_conv" 
        elif layer_name_new == "bn_conv1": 
             layer_name_new = "conv1_bn" 
        elif layer_name_new == "scale_conv1": 
             layer_name_new = "conv1_bn"  
        elif layer_name_new == "fc1000": 
             layer_name_new = "output" 
            
        print "Param len %d." % len(params) 
        param_num += len(params)
        for i in range(len(params)):
            if "bn" in layer_name:
                file = os.path.join(output_path, '_%s.w%s' %(layer_name_new, str(i + 1)))
            elif "scale" in layer_name or "fc" in layer_name or 'output' in layer_name: 
                suffix = "0" if i ==0 else "bias" 
                file = os.path.join(output_path, '_%s.w%s' %(layer_name_new, suffix))
            else: 
                file = os.path.join(output_path, '_%s.w%s' %(layer_name_new, str(i)))
            print "loading for layer %s." %layer_name
            print "loading for layer %s." %file
            if 'bn.w3' in file:
                data = np.asarray([1])
            else:
                data = load_parameter(file)
            print data.shape
            print params[i].shape
            data = np.reshape(data, params[i].shape)
            reference = np.array(params[i].data)
            if "fc" in layer_name or 'output' in layer_name:
                data = np.reshape(data, reference.T.shape)
                data = np.transpose(data)
            params[i].data[...] = data
    print "param_num = %d" % param_num
    net.save(conf.model_params)


def save_model_parameters(conf, layers, output_path):
    net = caffe.Classifier(conf.model_path, conf.model_params,
                       image_dims=(256, 256),
                       channel_swap=(2,1,0),
                       raw_scale=255)
    if len(layers) == 0: layers = net.params.keys()
    param_num = 0
    for layer_name in layers:
        params = net.params[layer_name]
        layer_name_new = layer_name.replace('/', '-')
        print layer_name
        print layer_name_new
        dict = {"a": "1", "b" : "2", "c" : "3", "d" : "4", "e" : "5", "f" : "6"}
        if "res" in layer_name_new:
            str1, str2 = layer_name_new.split("_") 
            layer_name_new = str1[0:4] + "_" + dict[str1[4]] + "_" + str2 + "_conv" 
        elif "bn" in layer_name_new and "branch" in layer_name_new: 
            str1, str2 = layer_name_new.split("_") 
            layer_name_new = "res" + str1[2] + "_" + dict[str1[3]] + "_" + str2 + "_bn"
        elif "scale" in layer_name_new and "branch" in layer_name_new: 
            str1, str2 = layer_name_new.split("_") 
            layer_name_new = "res" + str1[5] + "_" + dict[str1[6]] + "_" + str2 + "_bn"
        elif layer_name_new == "conv1": 
             layer_name_new = "conv1_conv" 
        elif layer_name_new == "bn_conv1": 
             layer_name_new = "conv1_bn" 
        elif layer_name_new == "scale_conv1": 
             layer_name_new = "conv1_bn"  
        elif layer_name_new == "fc1000": 
             layer_name_new = "output" 
            
        print "Param len %d." % len(params) 
        param_num += len(params)
        for i in range(len(params)):
            data = np.array(params[i].data)
            if "bn" in layer_name:
                file = os.path.join(output_path, '_%s.w%s' %(layer_name_new, str(i + 1)))
            elif "scale" in layer_name or "fc" in layer_name or "output" in layer_name: 
                suffix = "0" if i ==0 else "bias"
                file = os.path.join(output_path, '_%s.w%s' %(layer_name_new, suffix))
            else: 
                file = os.path.join(output_path, '_%s.w%s' %(layer_name_new, str(i)))
            print "Saving for layer %s." %layer_name
            print "Saving for layer %s." %file
            print data.shape, data.size
            if "fc" in layer_name or "output" in layer_name: 
                data = np.transpose(data)
            write_parameter(file, data.flatten())
    print "param_num = %d" % param_num

def write_parameter(outfile, feats):
    version = 0
    value_size  = 4;
    ret = ""
    for feat in feats:
        ret += feat.tostring()
    size = len(ret) / 4
    fo = open(outfile, 'wb')
    fo.write(struct.pack('iIQ', version, value_size, size))
    fo.write(ret)

def load_parameter(file):
    with open(file,'rb') as f:
        f.read(16)
        return np.fromfile(f, dtype=np.float32)

if __name__ == "__main__":
    conf = ModelConfig('caffe_model/animal.prototxt', 'caffe_model/animal.caffemodel')
    load_model_parameters(conf, [], 'paddle_model')
    pass

