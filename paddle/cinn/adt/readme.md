### 一、如何运行

#### 1.1 开启 FLAGS_cinn_enable_map_expr:

```
export FLAGS_enable_pir_api=1
export FLAGS_prim_all=True
export FLAGS_cinn_enable_map_expr=True
export GLOG_v=1
```

#### 1.2 执行 python 脚本:

```
cd test/ir/pir/cinn/adt && python test_cinn_sub_graph_map_expr.py
```

### 二、输出预览

#### 2.1 简单示例：以 Tensor x 为输入，执行 sin 和 relu 两个算子

```
builder = NetBuilder("MapExprTest")
x = builder.create_input(Float(32), inputs["x"].shape, "x")
y = builder.sin(x)
out = builder.relu(y)
```

#### 2.2 输出结果（随项目开发可能有变化）

```
fill_constant_1_sin_0_max_2(&t_var_1, t_x) {
  AnchoredMapStmt(t_var_0) {
    MapStmt(blockIdx.x=0..1, threadIdx.x=0..64) {
      fill_constant(&t_zero);
      sin(&t_var_0, t_x);
      max(&t_var_1, t_var_0, t_zero);
    }
  }
}
```

#### 2.3 各字段含义

| 字段  | 含义  |
| :------------ | :------------ |
| fill_constant_1_sin_0_max_2(&t_var_1, t_x)  |  MapExpr 名称为 fill_constant_1_sin_0_max_2（即当前 group 对应的 group_id），该 MapExpr 以 t_var_1 为输出，t_x 为输入，&为输出 Tensor 标识符|
| AnchoredMapStmt(t_var_0)  | 以 t_var_0 为 AnchorTensor 的一系列 Stmt，从 t_var_0 的下标索引可以推断出 Stmt 内所有其他 Tensor 的下标  |
| MapStmt(blockIdx.x=0..1, threadIdx.x=0..64)  | MapStmt 内所有 op 遵循如下调度策略：blockIdx.x 的取值为从 0 到 1，threadIdx.x 的取值为从 0 到 64 |
| fill_constant(&t_zero) | fill_constant 算子的输出 Tensor 为 t_zero |
| sin(&t_var_0, t_x) | sin 算子的输出 Tensor 为 t_var_0，输入 Tensor 为 t_x |
| max(&t_var_1, t_var_0, t_zero) | max 算子的输出 Tensor 为 t_var_1，输入 Tensor 为 t_var_0 和 t_zero |
