# Python Interface Design

## Variable as variable data type (not parameter)
The class Variable is introduced to define the variables that are not parameters.

All the variables are defined in a variable scope, `VarScope`, it will check duplicate variable names.

```python
class VarScope(object):
    '''
    VarScope is names' container of VarDesc in compile period.
    '''
    pass
```

The `Variable`'s definition is as follows
```python
class _unique_name_generator:
    counter = 0

    def __call__(self, prefix):
        name = '%s-%d' % (prefix, counter)
        counter += 1
        return name


unique_name_generator = _unique_name_generator()

class Variable(object):
    def __init__(self,
                 shape,
                 name=None,
                 prefix=None,
                 var_scope=None,
                 op=None,
                 reuse=False):
        '''
        var_scope is VarDesc's scope.
        '''
        self.shape = shape
        if name is None:
            if prefix is not None:
                name = unique_name_generator(prefix)
            else:
                name = unique_name_generator("unknown")
        self.name = name
        self.var_scope = var_scope
        if not reuse:
            assert self.name not in self.var_scope, "variable name %s duplicate in var scope" % self.name
        self.op = op
```

in above example, the `unique_name_generator` is introduced to make unique name with a prefix and can be used to generate variables' and operators' names.

## VarScope Stack for Block Inherience
Each block should has a variable scope that is inheriented from parent's variable scope.
That needs a stack of variable scopes, so that when user's model definition jumps out a sub-block, 
the following variables could be defined in current variable scope.

For example, when using a RNNOp

```python
# v is output of some op
v = someop()

rnn = RNNOp()
with rnn.stepnet():
    # current in a sub-block
    # all the variables are stored in a sub variable scope
    c = pd.fc(v)

e = someop()
```

first, the `v` variable is defined in global variable scope, 
while `c` is in rnn's sub-variable scope, 
the `e` variable is defined outside the rnn's stepnet, and is registered in global scope again.

Obviously, a variable scope is needed to help switch variale scopes.

There is another important issue in above code snippet, the `pd.fc` used inside rnn's stepnet will create parameters in some variable scope, 
currently, the trainable parameters should be registered in global scope so that the gradient optimizers can find and update them.

So, the parameters should only created in the head of the variable stack, we can implement this stack like

```python
g_varscope_stack = []


class VarScope(object):
    def __enter__(self):
        g_varscope_stack.append(self)

    def __exit__(self, type, value, traceback):
        g_varscope_stack.pop()

if not g_varscope_stack: g_varscope_stack.append(VarScope())

def Parameter(...):
    g_varscope_stack[0].register(...)
```
