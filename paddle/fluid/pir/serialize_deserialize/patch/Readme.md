# PIR下Save/load 版本兼容说明
## 一、yaml 配置说明
### op_patches：
对于op的补丁全部放在op_patches列表中，以op_name为标识，actions为具体操作列表，action为具体操作名称，object为具体操作对象。
```yaml
op_patches:
  - op_name : pd_op.xxx # op_name为标识
    actions:             # 补丁操作列表
      - action : xxxx # 具体操作名称
        object : xxxx  # 具体操作对象

      - action : xxxx  # 可能会对一个op进行多次操作
        object : xxxx
        type : xxxx
        default : xxx

  - op_name : builtin.xxx # 对另一个op进行操作
    actions :
      - action : xxxx

  - op_name : pd_op.xxx
    actions :
      - action : xxxx
        object : xxx
```
#### Attribute / OpresultAttribute 增删改：
- 增/改：
  Attribute与OpresultAttribute的新增和修改格式类似，需要指定object，type，default

```yaml
op_patches:
  - op_name : pd_op.data
    actions:
      - action : modify_output_attr # 修改OpresultAttribute
        object : stop_gradient      # 修改的具体属性名为stop_gradient
        type : pir::ArrayAttribute  # 修改属性类型为ArrayAttribute
        default :                   # 修改属性为具体值
          - type: pir::BoolAttribute  # ArrayAttribute类型需要对每一个字元素标识类型和值
            default: "false"
      - action : modify_attr        # 修改Attribute，与修改OpresultAttribute类似
        object : name
        type : pir::StrAttribute
        default : "B"
  - op_name : builtin.parameter
    actions :
      - action : add_attr           # 新增Attribute
        object : new_attribute      # 新增属性名为new_attribute
        type : pir::StrAttribute    # 新增属性类型为StrAttribute
        default : "new.attribute"   # 新增属性值为"new.attribute"
      - action : add_output_attr    # 新增OpresultAttribute
        object : new_Attribute      # 新增属性名为new_output
        type : pir::Int64Attribute  # 新增属性类型为ArrayAttribute
        default : 1                 # 新增属性为具体值
```

- 删：Attribute与OpresultAttribute的删除格式类似，只需要指定需要删除的具体对象object即可

```yaml
  - op_name : pd_op.fetch
    actions :
      - action : delete_attr  # 删除Attribute
        object : col          # 删除属性名为col
```

#### OpOperand / OpResult 增删改：
- OpOperand 不需要修改
- OpResult Type 修改：
  OpResult Type的修改改与Attribute类似，需要指定object，type，default。

  对于 Opresult Type，由于每个输出有且仅有一个Type，因此不存在增删的情况。

  ```yaml
  op_patches:
  - op_name : pd_op.data
    actions:
      - action : modify_output_type # 修改Opresult Type
        object : 0                  # 修改第几个输出的Type
        type : pir::DenseTensorType  # 修改属性类型为DenseTensorType
        default : [pir::Float32Type,[-1,30],"NCHW",[],0]   # 修改属性为具体值

  ```
- OpOperand / OpResult 增删改：需要提到block层进行处理。

  - op输入的删和op输出的增对于program结构不产生影响。删除输入直接修改输入列表即可；增加输出获取当前block中的op id的最大值，在上面进行新增即可。
  - 单独op输入的增和单独op输出的删会造成网络结构错误，因此报错。
  - op作为组合出现时，可以在内部进行输出的删和输入的增，不影响外部结构。

  ```yaml
  op_patches:
    - op_name : pd_op.data
      actions :
        - action : add_output      # 增加输出
          object : 1               # 增加为第几个输出
          type : pir::DenseTensorType  # 修改属性类型为DenseTensorType
          default : [pir::Float32Type,[-1,30],"NCHW",[],0]   # 修改属性为具体值
    - op_name : pd_op.add
      actions :
        - action : delete_input      # 删除输入
          object : 0                 # 删除第几个输入

  op_pair_patches:                   # 输入输出作为组合进行修改
    - op_pair : [pd_op.full, pd_op.full_like]
      actions:
        - action : add_value         # 增加输入和输出
          object : [1,2]             # 增加为第1个op的第几个输入，第2个op的第几个输出
          type : pir::DenseTensorType
          default : [pir::Float32Type,[1],"NCHW",[],0]
        - action : delete_value      # 删除输入和输出
          object : [1, 2]            # 删除第1个op的第几个输入，第2个op的第几个输出
  ```

#### 修改Attribute：
- Attribute值的修改都是随着op进行的，此处不需要修改。
- 对于Attribute本身，可能会修改的是其名称。但由于存储使用其缩写name，因此只要缩写不变，名称的修改不会影响兼容性，只需要在反序列化阶段对应修改为修改后的名称即可。
- 还可能存在修改Attribute的类型。比如废弃掉Int32Attribute，全部改为Int64Attribute。
```yaml
attr_patches:
  - attr_name : Int32Attribute   # 修改的attribute名称
    actions:
      - action : modify_name       # 修改Attribute名称
        type : Int64Attribute    # 修改为Int64Attribute
```
#### 修改Type：
- Type的修改与Attribute类似，值的修改都是随着OpResult进行的，此处不需要修改。
- Type修改名称也与上述情况类似。
- 存在废弃某一Type类型，使其更改为其他类型的情况。
- 对于某些type类型，其内部可能存在多个属性，因此需要指定修改的属性名称。
```yaml
type_patches:
  - type_name : Int32Type         # 修改的Type名称
    actions:
      - action : modify_name          # 修改Type名称
        type : Int64Type           # 修改为Int64Type
  - type_name : DenseTensorType    # 修改的Type名称
    actions:
      - action : add_type_attr           # 新增Type属性
      - object : 5                       # 新增属性为第几个属性
      - type : pir::Int64Attribute       # 新增属性类型为Int64Attribute
      - default : 0                      # 新增属性默认值
```

## pir_version 配置说明
### C++端版本号管理与CMake配置
- 版本号管理在C++端，在CMakeList.txt中配置。
- PIR版本号定义PIR的版本迭代，版本号与yaml文件名强相关。每次PIR进行更新并新增patch文件后，patch文件名顺序递增，版本号同时顺序递增。与Paddle的主版本号解耦，可以独立迭代。
  ```cmake
  # change pir version when new patches are added
  add_definitions(-DDEVELOP_VERSION=1)
  add_definitions(-DRELEASE_VERSION=1)
  ```

  ```tree
  ├─patch
  │  ├─0.yaml
  │  └─1.yaml
  ```
  - RELEASE_VERSION 为已发布的版本中PIR版本号，即为patch yaml文件名的最大值。
  - DEVELOP_VERSION 为当前develop分支下的PIR版本号，若存在未发布的新增patch，配置在`0.yaml`中，且当前的develop pir 版本号为0。

- ReadModule和WriteModule参数中的pir_version设为默认值，可以不用传递。pir_version 函数默认值为-1，进入函数后会获取CMake中配置的当前的PIR版本号。

### Python端
- Paddle的主版本号定义在Python端，与PIR version不产生关联。Python端不再需要获取和传入pir_version，直接使用默认值即可。
### Paddle发版要求
- 需要确认Paddle发版时develop版本被修改为正式版本，即若Python端的版本号不为0.0.0，则pir_version不能为0。
