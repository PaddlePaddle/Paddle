import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator
import numpy
import paddle.v2 as paddle
import time

BATCH_SIZE = 100
numpy.random.seed(100)
def create_var(name, m, n, place, scope):
    var = scope.new_var(name);
    tensor = var.get_tensor()
    if m == 1:
        tensor.set_dims([n]);
        tensor.alloc_float(place)
        #data = numpy.ones((n))
        data = numpy.random.rand(n) - 0.5
        tensor.set(data, place)
    else:
        tensor.set_dims([m,n])
        tensor.alloc_float(place)
        #data = numpy.ones((m, n))
        data = numpy.random.rand(m, n) - 0.5
        tensor.set(data, place)
    #data = numpy.random.ran((m, n))

     
scope = core.Scope()
#place = core.CPUPlace()
place = core.FPGAPlace(0)
# if you want to test GPU training, you can use gpu place
# place = core.GPUPlace(0)
dev_ctx = core.DeviceContext.create(place)

batch = 300
feat_dim = 192;
hidden_1 = 512;
hidden_2 = 512;
hidden_3 = 512;
hidden_4 = 64;

create_var("input", batch, feat_dim, place, scope)
create_var("w_1", feat_dim, hidden_1, place, scope)
create_var("out_1", batch, hidden_1, place, scope)

create_var("w_2", hidden_1, hidden_2, place, scope)
create_var("out_2", batch, hidden_2, place, scope)

create_var("w_3", hidden_2, hidden_3, place, scope)
create_var("out_3", batch, hidden_3, place, scope)

create_var("w_4", hidden_3, hidden_4, place, scope)
create_var("out_4", batch, hidden_4, place, scope)

opList = []

opList.append(Operator("mul", X="input", Y="w_1", Out="out_1"))
opList.append(Operator("mul", X="out_1", Y="w_2", Out="out_2"))
opList.append(Operator("mul", X="out_2", Y="w_3", Out="out_3"))
opList.append(Operator("mul", X="out_3", Y="w_4", Out="out_4"))
   
start = time.time()
for i in range(1000):
    for op in opList:
        op.run(scope, dev_ctx)
    cost_data = numpy.array(scope.find_var("out_4").get_tensor())
end = time.time()

print "duration:"
print end - start
numpy.set_printoptions(precision=4)
print cost_data

