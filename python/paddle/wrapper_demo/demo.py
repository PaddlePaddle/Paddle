# all the layer, op namespace are imported into pd for short
# the old absolute path should also work
# for example, pd.data -> paddle.layer.data
import paddle.v2 as pd

images = pd.data(name='pixe', type=pd.dense_vector(784))

label = pd.data(name='label', type=pd.integer_value(10))

prediction = pd.fc(input=images, size=10, act=...)
cost = pd.cross_entropy(prediction, label)

optimizer = pd.Momentum(...)

# following are different styles should supported

# v2 style, seems that hard to support multiple sub-model
parameters = pd.parameters.create(cost)
trainer = pd.SGD(cost=[cost], parameters=parameters, update_equation=optimizer)
trainer.train(reader=..., num_passes=5)

# style with new features borrowed from tf and pytorch
trainer = pd.SGD()
trainer.init_parameters()
# train a sub-model if there has more than one sub-models has different costs like gan
trainer.train(targets=[cost], update_equation=optimizer, reader=...)

# just forward run is supported
# borrowed from tf, forward run a sub-model whose end point is targets
#trainer.run(targets=[cost], reader=...)
