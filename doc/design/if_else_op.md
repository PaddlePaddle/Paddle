In an if_op, only inputs with condition satisfied will be run. The op could have multiple inputs and multiple outputs.
We should have the following design:

```python
# A 1-d bool vector
cond = Var()
# create an op
if = pd.if_op()

with if.true_block() as block:
  x1 = if.input(x1)
  x2 = if.input(x2)
  y = pd.add(x1, x2)
  y2 = pd.fc(x1) # contains (w,b)
  if.output(y)
  if.output(y2)
  
o1, o2 = if(cond)
```

In an if_op, only inputs with condition satisfied will be run.
We should have the following design:
```python
# A 1-d bool vector
cond = Var()
# create an op
if = pd.if_op()

with if.true_block() as block:
  x1 = if.input(x1)
  x2 = if.input(x2)
  y = pd.add(x1, x2)
  y2 = pd.fc(x1) # contains (w,b)
  if.output(y, name="y")
  if.output(y2, name="y2")

with if.false_block() as block:
  x1 = if.input(x1)
  x2 = if.input(x2)
  y = pd.fc(x2)
  y2 = pd.softmax(x1) 
  if.output(y, name="y")
  if.output(y2, name="y2")
  
o1, o2 = if(cond)
```

Some questions:
 1. how to know which inputs will be selected by condition?
 e.g. True_block():
  y = pd.fc(x)
  # we will have x, w, b all as inputs
  # but only x will be selected by cond, how can the block know?

