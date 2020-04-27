# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Sentiment Classification in Paddle Dygraph Mode. """


from __future__ import print_function
import numpy as np
import paddle.fluid as fluid
from hapi.model import set_device, Model, CrossEntropy, Input
from hapi.configure import Config
from hapi.text.senta import SentaProcessor
from hapi.metrics import Accuracy
from models import CNN, BOW, GRU, BiGRU
import json
import os

args = Config(yaml_file='./senta.yaml')
args.build()
args.Print()

device = set_device("gpu" if args.use_cuda else "cpu")
dev_count = fluid.core.get_cuda_device_count() if args.use_cuda else 1

def main():
    if args.do_train:
        train()
    elif args.do_infer:
        infer()

def train():
    fluid.enable_dygraph(device)
    processor = SentaProcessor(
        data_dir=args.data_dir,
        vocab_path=args.vocab_path,
        random_seed=args.random_seed)
    num_labels = len(processor.get_labels())

    num_train_examples = processor.get_num_examples(phase="train")

    max_train_steps = args.epoch * num_train_examples // args.batch_size // dev_count

    train_data_generator = processor.data_generator(
        batch_size=args.batch_size,
        padding_size=args.padding_size,
        places=device,
        phase='train',
        epoch=args.epoch,
        shuffle=False)

    eval_data_generator = processor.data_generator(
        batch_size=args.batch_size,
        padding_size=args.padding_size,
        places=device,
        phase='dev',
        epoch=args.epoch,
        shuffle=False)
    if args.model_type == 'cnn_net':
        model = CNN( args.vocab_size, args.batch_size,
                     args.padding_size)
    elif args.model_type == 'bow_net':
        model = BOW( args.vocab_size, args.batch_size,
                     args.padding_size)
    elif args.model_type == 'gru_net':
        model = GRU( args.vocab_size, args.batch_size,
                     args.padding_size)
    elif args.model_type == 'bigru_net':
        model = BiGRU( args.vocab_size, args.batch_size,
                       args.padding_size)
    
    optimizer = fluid.optimizer.Adagrad(learning_rate=args.lr, parameter_list=model.parameters()) 
    
    inputs = [Input([None, None], 'int64', name='doc')]
    labels = [Input([None, 1], 'int64', name='label')]
    
    model.prepare(
        optimizer,
        CrossEntropy(),
        Accuracy(topk=(1,)),
        inputs,
        labels,
        device=device)
    
    model.fit(train_data=train_data_generator,
              eval_data=eval_data_generator,
              batch_size=args.batch_size,
              epochs=args.epoch,
              save_dir=args.checkpoints,
              eval_freq=args.eval_freq,
              save_freq=args.save_freq)

def infer():
    fluid.enable_dygraph(device)
    processor = SentaProcessor(
        data_dir=args.data_dir,
        vocab_path=args.vocab_path,
        random_seed=args.random_seed)

    infer_data_generator = processor.data_generator(
        batch_size=args.batch_size,
        padding_size=args.padding_size,
        places=device,
        phase='infer',
        epoch=1,
        shuffle=False)
    if args.model_type == 'cnn_net':
        model_infer = CNN( args.vocab_size, args.batch_size,
                           args.padding_size)
    elif args.model_type == 'bow_net':
        model_infer = BOW( args.vocab_size, args.batch_size,
                           args.padding_size)
    elif args.model_type == 'gru_net':
        model_infer = GRU( args.vocab_size, args.batch_size,
                           args.padding_size)
    elif args.model_type == 'bigru_net':
        model_infer = BiGRU( args.vocab_size, args.batch_size,
                             args.padding_size)
    
    print('Do inferring ...... ')
    inputs = [Input([None, None], 'int64', name='doc')]
    model_infer.prepare(
        None,
        CrossEntropy(),
        Accuracy(topk=(1,)),
        inputs,
        device=device)
    model_infer.load(args.checkpoints, reset_optimizer=True)
    preds = model_infer.predict(test_data=infer_data_generator)
    preds = np.array(preds[0]).reshape((-1, 2))

    if args.output_dir:
        with open(os.path.join(args.output_dir, 'predictions.json'), 'w') as w:
            
            for p in range(len(preds)):
                label = np.argmax(preds[p])
                result = json.dumps({'index': p, 'label': label, 'probs': preds[p].tolist()})
                w.write(result+'\n')
        print('Predictions saved at '+os.path.join(args.output_dir, 'predictions.json'))

if __name__ == '__main__':
    main()
