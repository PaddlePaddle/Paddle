# FeedOp and FetchOp Design Doc

### Motivation

Python programer needs an interface to feed the data to PaddlePaddle, run the model, and fetch the result from it. Since PaddlePaddle runtime only goes through a graph of ops, we need to design corresponding Ops and add them to the graph.

### Challenge

1. During the runtime of a particular Op, it only knows which `Variable` to be read from and written to. It doesn't have a direct access to python object.
2. I/O involves copying data between python object and C++ object. In the current design, python only passes `ProgramDesc` to C++.

### Solution

To solve the first challenge, we add two **global** `Variable`s, `feed_result` and `fetch_result`. The operator can read/write these variables and later on it can be exchanged with python interface. The content of each is a `map<string, LoDTensor>`, where `string` is the `variable.name` and `LoDTensor` is the actual data. Be aware that the actually data is always on CPU.

To solve the second challenge, we add two python interface `feed_value` and `fetch_value`. 

```Python
def feed_value(variable, np_variable):
    """Overwrite feed_result[variable.name] with a numpy.array
	
    Args:
    	variable:     Paddle Variable to be overwritten
    	np_variable:  numpy_array to write the tensor feed_result[variable.name]
    """

def fetch_value(variable):
    """Fetch the value of fetch_result[variable.name]
       and convert it into a numpy.array
    
    Args:
    	variable:     A Paddle Variable to be fetched
    
    Returns:
    	numpy_array:  Value of the fetched variable
    """
```

Now, we are able to implement the `FeedOp` and `FetchOp` as follows.

```c++
void FeedOp::Compute(const ExecutionContext& context)
{
  // Get Tensor reference in feed_result
  string name = ctx.Output<Tensor>("Output")->name();
  auto& var = GetScope()->GetVar("feed_result");
  auto& input_tensor = var->Get<map<string, LoDTensor>>[name];
  
  // Memcopy from feed_result to output
  ctx.Output<Tensor>("Output")->mutable_data<T>(ctx.GetPlace());
  ctx.Output<Tensor>("Output")->CopyFrom(input_tensor, ctx.GetPlace()); 
}

void FetchOp::Compute(const ExecutionContext& context)
{
  // Get Tensor reference in fetch_result
  string name = ctx.Output<Tensor>("Input")->name();
  auto& var = GetScope()->GetVar("fetch_result");
  auto& output_tensor = var->Get<map<string, LoDTensor>>[name];

  // Memcopy from input to fetch_result
  output_tensor->mutable_data<T>(CPUPlace);
  output_tensor->CopyFrom(ctx.Output<Tensor>("Input"), CPUPlace);
}
```

### Example

```python
# Build the model -------------------
data = Variable(dim)
label = Variable(label_dim)

w = Parameter()				
x = mul(w, data)
cost = cross_entropy(x, label)		
opts = optimize(cost, variable_to_be_optimized)

# Initialization -------------------
if is_load_from_checkpoint:
    executor.run(load_checkpoint)
else
    executor.run(init_parameter)

# Run -------------------
while not converge:
    # user loads data
    np_data, np_label = load_input_data()
    
    # user defines the maping
    my_feed_dict = {data: np_data, label: np_label}

    # Case 1: for the foward pass.
    np_cost = executor.run(target=cost, feed_dict=my_feed_dict)
    # Case 2: for the backward passing.
    np_cost, np_opts = executor.run(target=[cost, opts], feed_dict=my_feed_dict)
```

Under the hood:

```Python
def executor.run(target, feed_dict):
    """Overwrite the feed_dict and evalate the target
	
    Args:
    	target:			A list of Paddle Variable to be evaluated
    	feed_dict:		A dictionary of {variable : numpy_array} to be overwritten
    """
    for var, np_var in feed_dict:
        feed_value(var, np_var) # do memcpy to feed_result
        feed_op(var)            # add feed_op into the graph

    for var in target:
        fetch_op(var)           # add feed_op into the graph
	
    executor.prune()            # get a subgraph
    executor.run_graph()        # actually run the graph
    
    return [fetch_value(var) for var in target] # do memcpy from fetch_result
```
