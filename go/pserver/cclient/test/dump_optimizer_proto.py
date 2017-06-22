import OptimizerConfig_pb2 as pb

config = pb.OptimizerConfig()
config.clip_norm = 0.1
config.lr_policy = pb.OptimizerConfig.Const
config.optimizer = pb.OptimizerConfig.SGD
config.sgd.momentum = 0.0
config.sgd.decay = 0.0
config.sgd.nesterov = False
config.const_lr.learning_rate = 0.1
s = config.SerializeToString()
with open("optimizer.pb.txt", 'w') as f:
  f.write(s)
