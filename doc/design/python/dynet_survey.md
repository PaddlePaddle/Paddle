## dynet设计

1 有一个全局的model

用户的参数需要从model中创建，这样才能保证model能记录到parameter

```
Model model;
Parameter pW = model.add_parameters({1})
```



2 有一个全局的trainer，负责参数更新，trainer的一个参数是Model

```
SimpleSGDTrainer trainer(model, 0.1);

```

3 有一个全局的graph

- 用户通过书写Expression构建graph，接口以全局函数的形式提供

- input和parameter都是graph中的节点，分别提供parameter方法来实现添加parameter节点和input方法来实现添加input节点

- 在parameter和input的全局函数中，需要指定全局的graph

- 返回的Expression中会记录graph，后续添加节点就直接从输入expression拿就可以了

```
ComputationGraph cg;
Expression W = parameter(cg, pW);

Expression in = input(cg, xs[i]);
Expression label = input(cg, ys[i]);
Expression pred = W * in;
Expression loss = square(pred - label);

```

parameter/input/square等都是全局方法，返回值是Expression，可以看一下Expression的定义

```
struct Expression {
  ComputationGraph* cg;
  VariableIndex i;
  unsigned graph_id;
  
  Expression(Computation* cg, VariableIndex i) : pg(pg), i(i), graph_id(pg->get_id()) {}

};
```

graph至此构建完毕，dynet的反向过程并没有显示的拿出来



4 graph执行

```
cg.forward(loss);
cg.backward(loss);
trainer.update();
```

graph执行依次调用forward和backward方法，参数更新不是graph的一部分，由trainer来执行



## 启发与讨论

### 启发

1. 需要有一个全局的Model管理所有的parameter，Model提供一个create_parameter的接口用于创建parameter

2. 需要有一个全局的Net，来记录网络的拓扑结构。Net是否是Model的成员变量都可以

3. 参数更新也是Net的一部分，

### 讨论

1. 输入数据与参数是不是graph的节点

2. 输入数据怎么加载，使用LoadOp，还是创建Variable，然后直接FeedVariable

3. 参数怎么初始化，怎么加载

有一部分参数是随机产生的，这部分参数是由paddle提供的Op产生；

还有一些参数是从用户给定数据加载的，使用LoadOp，还是FeedVariable
