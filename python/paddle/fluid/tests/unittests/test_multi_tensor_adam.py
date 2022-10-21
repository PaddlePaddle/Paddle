import unittest
import paddle
import copy
import paddle.nn as nn
from copy import deepcopy
import numpy as np

class MLPLayer(nn.Layer):
   def __init__(self, input_size, hidden_size, output_size, n):
        super(MLPLayer, self).__init__()
        self.linear_first = nn.Linear(input_size, hidden_size)
        self.linear_mid = nn.Sequential(('l0', nn.Linear(hidden_size, hidden_size)))
        for i in range(n - 1):
            self.linear_mid.add_sublayer('l' + str(i+1), nn.Linear(hidden_size, hidden_size))
        self.linear_last = nn.Linear(hidden_size, output_size)

   def forward(self, x):
        x = self.linear_first(x)
        x = self.linear_mid(x)
        x = self.linear_last(x)
        return x.mean()

class TestMultiTensorAdam(unittest.TestCase):
    
    def setUp(self):
        paddle.disable_static()
        self.input_size = 800
        self.hidden_size = 500
        self.output_size = 700
        self.n = 10
        
    def get_multi_tensor_adam(self, mode):
    
        paddle.seed(10)
        
        model = MLPLayer(self.input_size, self.hidden_size, self.output_size, self.n)
        
        inp = paddle.uniform([10, self.input_size], dtype="float32", seed = 10)
        
        out = model(inp)

        beta1 = paddle.to_tensor([0.9], dtype="float32")
        beta2 = paddle.to_tensor([0.99], dtype="float32")

        opt = paddle.optimizer.MultiTensorAdam(
                        learning_rate=0.1,
                        parameters=model.parameters(),
                        weight_decay=0.01,
                        beta1=beta1,
                        beta2=beta2,
                        mode= mode)
        out.backward()
        opt.step()
        opt.clear_grad()
        
        return model.parameters()

    def get_adam_or_adamw(self, mode):

        paddle.seed(10)

        model = MLPLayer(self.input_size, self.hidden_size, self.output_size, self.n)
        
        inp = paddle.uniform([10, self.input_size], dtype="float32", seed = 10)
        
        out = model(inp)
        
        beta1 = paddle.to_tensor([0.9], dtype="float32")
        beta2 = paddle.to_tensor([0.99], dtype="float32")

        if not mode:
            opt = paddle.optimizer.Adam(
                            learning_rate=0.1,
                            parameters=model.parameters(),
                            weight_decay=0.01,
                            beta1=beta1,
                            beta2=beta2)
        else:
            opt = paddle.optimizer.AdamW(
                            learning_rate=0.1,
                            parameters=model.parameters(),
                            weight_decay=0.01,
                            beta1=beta1,
                            beta2=beta2)
        
        out.backward()
        opt.step()
        opt.clear_grad()
        
        return model.parameters()

    def get_multi_tensor_adam_dict(self, mode):
        
        paddle.seed(10)
        
        model = MLPLayer(self.input_size, self.hidden_size, self.output_size, self.n)
        
        inp = paddle.uniform([10, self.input_size], dtype="float32", seed = 10)
        
        out = model(inp)

        beta1 = paddle.to_tensor([0.9], dtype="float32")
        beta2 = paddle.to_tensor([0.99], dtype="float32")
        
        paramters_dict_list=[]
        i = 0
        for param in model.parameters():
            paramters_dict_list.append({'params': param, 
                                        'weight_decay': 0.001 * i,
                                        'learning_rate': 0.01 * i,
                                        'beta1': 0.01 * i})
            i = i + 1
            

        opt = paddle.optimizer.MultiTensorAdam(
                        learning_rate=0.1,
                        parameters=paramters_dict_list,
                        weight_decay=0.01,
                        beta1=beta1,
                        beta2=beta2,
                        mode= mode)
        out.backward()
        opt.step()
        opt.clear_grad()
        
        return model.parameters()

    def get_adam_or_adamw_dict(self, mode):

        paddle.seed(10)

        model = MLPLayer(self.input_size, self.hidden_size, self.output_size, self.n)
        
        inp = paddle.uniform([10, self.input_size], dtype="float32", seed = 10)
        
        out = model(inp)
        
        beta1 = paddle.to_tensor([0.9], dtype="float32")
        beta2 = paddle.to_tensor([0.99], dtype="float32")
        
        paramters_dict_list=[]
        i = 0
        for param in model.parameters():
            paramters_dict_list.append({'params': param,
                                        'weight_decay': 0.001 * i,
                                        'learning_rate': 0.01 * i,
                                        'beta1': 0.01 * i})
            i = i + 1

        if not mode:
            opt = paddle.optimizer.Adam(
                            learning_rate=0.1,
                            parameters=paramters_dict_list,
                            weight_decay=0.01,
                            beta1=beta1,
                            beta2=beta2)
        else:
            opt = paddle.optimizer.AdamW(
                            learning_rate=0.1,
                            parameters=paramters_dict_list,
                            weight_decay=0.01,
                            beta1=beta1,
                            beta2=beta2)
        
        out.backward()
        opt.step()
        opt.clear_grad()
        
        return model.parameters()
    
    def get_multi_tensor_adam_fp16(self, mode):
    
        paddle.seed(10)
        np.random.seed(10)
        
        model = MLPLayer(self.input_size, self.hidden_size, self.output_size, self.n)
        model = paddle.amp.decorate(models=model, level='O2')
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        
        inp = np.random.random((10, self.input_size)).astype("float16")
        inp = paddle.to_tensor(inp)

        beta1 = paddle.to_tensor([0.9], dtype="float32")
        beta2 = paddle.to_tensor([0.99], dtype="float32")

        opt = paddle.optimizer.MultiTensorAdam(
                        learning_rate=0.1,
                        parameters=model.parameters(),
                        weight_decay=0.01,
                        beta1=beta1,
                        beta2=beta2,
                        mode= mode)
        
        with paddle.amp.auto_cast(level='O2'):
            out = model(inp)
            loss = paddle.mean(out)
        scaled = scaler.scale(loss)
        scaled.backward()
        scaler.step(opt)
        opt.clear_grad()
        
        return model.parameters()
    
    def get_adam_or_adamw_fp16(self, mode):

        paddle.seed(10)
        np.random.seed(10)

        model = MLPLayer(self.input_size, self.hidden_size, self.output_size, self.n)
        model = paddle.amp.decorate(models=model, level='O2')
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        
        inp = np.random.random((10, self.input_size)).astype("float16")
        inp = paddle.to_tensor(inp)
        
        beta1 = paddle.to_tensor([0.9], dtype="float32")
        beta2 = paddle.to_tensor([0.99], dtype="float32")

        if not mode:
            opt = paddle.optimizer.Adam(
                            learning_rate=0.1,
                            parameters=model.parameters(),
                            weight_decay=0.01,
                            beta1=beta1,
                            beta2=beta2)
        else:
            opt = paddle.optimizer.AdamW(
                            learning_rate=0.1,
                            parameters=model.parameters(),
                            weight_decay=0.01,
                            beta1=beta1,
                            beta2=beta2)
            
        with paddle.amp.auto_cast(level='O2'):
            out = model(inp)
            loss = paddle.mean(out)
        scaled = scaler.scale(loss)
        scaled.backward()
        scaler.step(opt)
        opt.clear_grad()
        
        return model.parameters()
    
    def get_multi_tensor_adam_dict_fp16(self, mode):
        
        paddle.seed(10)
        np.random.seed(10)
        
        model = MLPLayer(self.input_size, self.hidden_size, self.output_size, self.n)
        model = paddle.amp.decorate(models=model, level='O2')
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        
        inp = np.random.random((10, self.input_size)).astype("float16")
        inp = paddle.to_tensor(inp)

        beta1 = paddle.to_tensor([0.9], dtype="float32")
        beta2 = paddle.to_tensor([0.99], dtype="float32")
        
        paramters_dict_list=[]
        i = 0
        for param in model.parameters():
            paramters_dict_list.append({'params': param, 
                                        'weight_decay': 0.001 * i,
                                        'learning_rate': 0.01 * i,
                                        'beta1': 0.01 * i})
            i = i + 1
            

        opt = paddle.optimizer.MultiTensorAdam(
                        learning_rate=0.1,
                        parameters=paramters_dict_list,
                        weight_decay=0.01,
                        beta1=beta1,
                        beta2=beta2,
                        mode= mode)
        with paddle.amp.auto_cast(level='O2'):
            out = model(inp)
            loss = paddle.mean(out)
        scaled = scaler.scale(loss)
        scaled.backward()
        scaler.step(opt)
        opt.clear_grad()
        
        return model.parameters()
    
    def get_adam_or_adamw_dict_fp16(self, mode):

        paddle.seed(10)
        np.random.seed(10)

        model = MLPLayer(self.input_size, self.hidden_size, self.output_size, self.n)
        model = paddle.amp.decorate(models=model, level='O2')
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        
        inp = np.random.random((10, self.input_size)).astype("float16")
        inp = paddle.to_tensor(inp)
        
        beta1 = paddle.to_tensor([0.9], dtype="float32")
        beta2 = paddle.to_tensor([0.99], dtype="float32")
        
        paramters_dict_list=[]
        i = 0
        for param in model.parameters():
            paramters_dict_list.append({'params': param,
                                        'weight_decay': 0.001 * i,
                                        'learning_rate': 0.01 * i,
                                        'beta1': 0.01 * i})
            i = i + 1

        if not mode:
            opt = paddle.optimizer.Adam(
                            learning_rate=0.1,
                            parameters=paramters_dict_list,
                            weight_decay=0.01,
                            beta1=beta1,
                            beta2=beta2)
        else:
            opt = paddle.optimizer.AdamW(
                            learning_rate=0.1,
                            parameters=paramters_dict_list,
                            weight_decay=0.01,
                            beta1=beta1,
                            beta2=beta2)
        
        with paddle.amp.auto_cast(level='O2'):
            out = model(inp)
            loss = paddle.mean(out)
        scaled = scaler.scale(loss)
        scaled.backward()
        scaler.step(opt)
        opt.clear_grad()
        
        return model.parameters()

    def run_multi_tensor_adam(self, mode):
        parameters = self.get_multi_tensor_adam(mode)
        parameters_1 = self.get_adam_or_adamw(mode)
        for i, j in zip(parameters, parameters_1):
            np.allclose(i.numpy(), j.numpy())
            
    def run_multi_tensor_adam_fp16(self, mode):
        parameters = self.get_multi_tensor_adam_fp16(mode)
        parameters_1 = self.get_adam_or_adamw_fp16(mode)
        for i, j in zip(parameters, parameters_1):
            np.allclose(i.numpy(), j.numpy())
    
    def run_multi_tensor_adam_dict(self, mode):
        parameters = self.get_multi_tensor_adam_dict(mode)
        parameters_1 = self.get_adam_or_adamw_dict(mode)
        for i, j in zip(parameters, parameters_1):
            np.allclose(i.numpy(), j.numpy())
            
    def run_multi_tensor_adam_dict_fp16(self, mode):
        parameters = self.get_multi_tensor_adam_dict_fp16(mode)
        parameters_1 = self.get_adam_or_adamw_dict_fp16(mode)
        for i, j in zip(parameters, parameters_1):
            np.allclose(i.numpy(), j.numpy())
            
    def test_main(self):
        paddle.set_device('gpu')
        for mode in [True, False]:
            self.run_multi_tensor_adam(mode)
            self.run_multi_tensor_adam_fp16(mode)
            self.run_multi_tensor_adam_dict(mode)
            self.run_multi_tensor_adam_dict_fp16(mode)
    

if __name__ == "__main__":
    unittest.main()