# DRR( Declarative Rewrite Rule) PASS用户使用手册
---
## 1. 相关背景

PASS是对IR进行优化的关键组件，而DAG-to-DAG的变换（将原图中的一个DAG子图替换成另一个DAG子图）是最常见的Pass类型。DAG-to-DAG的变换可以划分为匹配和重写两个步骤：匹配是根据已知子图在Program中完全匹配到对应的目标子图，重写是将匹配到的图结构替换为新的子图。

DRR ( Declarative Rewrite Rule ) 是来处理这种DAG-to-DAG类型的一套PASS组件。DRR 能降低PASS的开发成本，让开发者集中在对优化逻辑的处理上，而不需要关心底层IR的数据结构。开发者通过一套简洁易用的接口对目标子图和需要替换成的新子图进行模式声明后，DRR 就能自动的在 Program 中对原图进行匹配，并替换成新子图。

以消除冗余 CastOp 的 PASS 为例，使用 DRR 的代码开发示例如下：
~~~ c++
// 1. 继承DrrPatternBase的特化模板类
class RemoveRedundentCastPattern
    : public pir::drr::DrrPatternBase<RemoveRedundentCastPattern> {
  // 2. 重载operator()
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    // 3. 使用Op、Tensor和Attribute定义一个包含两个连续CastOp的SourcePattern
    auto pat = ctx->SourcePattern();

    // arg0是第一个CastOp的输入Tensor的ID
    //

    pat.Tensor("tmp") =                          // CastOp输出Tensor命名为"tmp"
        pat.Op(paddle::dialect::CastOp::name(),  // 传入CastOp的name
               {{"dtype", pat.Attr("dtype1")}})  // CastOp的"dtype"属性的对应的全局唯一ID为"dtype1"
               (pat.Tensor("arg0"));             // CastOp输入Tensor为"arg0"
    pat.Tensor("ret") =
        pat.Op(paddle::dialect::CastOp::name(),
               {{"dtype", pat.Attr("dtype2")}})(pat.Tensor("tmp"));
    // 4. 定义Constrain
    pat.RequireEqual(pat("tmp").dtype(), pat.Tensor("ret").dtype());

    // 5. 定义ResultPattern
    auto res = pat.ResultPattern();
    res.Tensor("ret") =
        res.Op(paddle::dialect::CastOp::name(),
               {{"dtype", pat.Attr("dtype2")}})(res.Tensor("arg0"));
  }
};
~~~

DRR PASS包含以下三个部分：
+ `SourcePattern`：用于描述在Program中待匹配的目标子图
+ `Constrains`：用于指定`SourcePattern`匹配的限制条件（非必需）
+ `ResultPattern`：用于描述需要替换为的模式子图
开发者只需要定义出`SourcePattern`, `Constrains`和`ResultPattern`即可实现一个完整的PASS。

**注意：**
**1. DRR仅支持对闭包（除Pattern输入输出Tensor以外，所有的内部Tensor不能被Pattern外部Op使用）的 SourcePattern 和 ResultPattern 进行匹配替换，若定义的Pattern在Program中不闭包则匹配失败**
**2. ResultPattern的输入输出需要满足是SourcePattern的输入输出的子集**
## 2. 接口列表

<table>
	 <tr>
		<th> 类 </th>
		<th> 函数 </th>
		<th> 功能描述 </th>
		<th> 参数解释 </th>
	 </tr>
	<tr>
		<td rowspan="1">DrrPatternBase</td>
		<td> <pre> virtual void operator()(
        pir::drr::DrrPatternContext* ctx) const </pre></td>
		<td> 实现DRR PASS的入口函数 </td>
		<td> ctx: 创建Patten所需要的Context参数</td>
	</tr>
	<tr>
		<td rowspan="6"> SourcePattern</td>
		<td><pre> const drr::Op& Op(
    const std::string& op_type,
    const std::unordered_map&lt;std::string, Attribute&gt;& attributes)</pre></td>
		<td> 在SourcePattern中定义一个Op</td>
		<td> op_type: 定义的op名称，可以通过paddle::dialect::xxOp
	::name()接口获取 <br> attributes : 所创建的Op的属性信息 </td>
	</tr>
	<tr>
		<td><pre> const drr::Tensor& Tensor(
        const std::string& tensor_name) </pre></td>
		<td> 在SourcePattern中定义一个名为tensor_name的tensor</td>
		<td>  tensor_name: 定义的Tensor的名称，需要满足SourcePattern内唯一 </td>
	</tr>
	<tr>
		<td> <pre> Attribute Attr(
        const std::string& attr_name) const </pre></td>
		<td> 在SourcePattern中定义一个名为attr_name的属性 </td>
		<td> attr_name: 属性的名称，需要满足SourcePattern内唯一 </td>
	</tr>
	<tr>
		<td><pre> void RequireEqual(
        const TensorShape& first,
        const TensorShape& second)</pre></td>
		<td> 要求SourcePattern中两个Tensor的TensorShape相同</td>
		<td> first: 第一个TensorShape <br> second : 第二个TensorShape</td>
	</tr>
		<tr>
		<td><pre> void RequireEqual(
        const TensorDataType& first,
        const TensorDataType& second)</pre></td>
		<td> 要求SourcePattern中两个Tensor的数据类型相同</td>
		<td> first: 第一个Tensor的DataType <br> second : 第二个Tensor的DataType</td>
	</tr>
	<tr>
		<td> <pre>void RequireNativeCall(
        const std::function&lt;bool(const MatchContext&)&gt;& custom_fn)</pre></td>
		<td> 在SourcePattern中定义一个约束，可以利用此接口和lamda表达式实现对SourcePattern的自定义约束</td>
		<td> custom_fn: 自定义的约束函数</td>
	</tr>
	<tr>
		<td rowspan="5"> ResultPattern</td>
				<td><pre> const drr::Op& Op(
    const std::string& op_type,
    const std::unordered_map&lt;std::string, Attribute&gt;&  attributes) </pre></td>
		<td> 在ResultPattern中定义一个Op </td>
		<td> op_type: 定义的op名称，可以通过paddle::dialect::xxOp
	::name()接口获取<br> attributes : 所创建的Op的属性信息 </td>
	</tr>
	<tr>
		<td> <pre>const drr::Tensor& Tensor(
        const std::string& tensor_name)</pre></td>
		<td> 在ResultPattern中定义一个名为tensor_name的tensor</td>
		<td> tensor_name: 定义的Tensor的名称，需要满足ResultPattern内唯一 </td>
	</tr>
	<tr>
		<td><pre>Attribute Attr(
        const std::string& attr_name) const </pre></td>
		<td> 在 ResultPattern 中定义一个名为 attr_name 的属性 </td>
		<td> attr_name: 属性的名称，需要满足ResultPattern内唯一 </td>
	</tr>
<tr>
		<td><pre>using AttrComputeFunc = std::function&lt;std::any(const MatchContext&)&gt;;
Attribute Attr(const AttrComputeFunc& attr_compute_func) const</pre></td>
		<td> 通过自定义的计算逻辑AttrComputeFunc，创建出一个Attribute</td>
		<td>attr_compute_func: 自定义的计算逻辑</td>
	</tr>
	<tr>
		<td> <pre>drr::Tensor& NoneTensor()</pre></td>
		<td> 当一个 Op的输入Tensor 是一个可选项并且不需要时，需要使用 NoneTensor 来占位</td>
		<td> / </td>
	</tr>
	<tr>
		<td rowspan="2"> TensorShape</td>
		<td><pre>explicit TensorShape(
        const std::string& tensor_name) </pre></td>
		<td> 抽象出来描述Tensor的shape的类 </td>
		<td> tensor_name: 被描述的Tensor的name </td>
	</tr>
	<tr>
		<td><pre> const std::string& tensor_name() const</pre></td>
		<td> 获取tensor的name</td>
		<td>  / </td>
	</tr>
	<tr>
		<td rowspan="2"> TensorDataType</td>
		<td><pre>explicit TensorDataType(
        const std::string& tensor_name)</pre></td>
		<td> 抽象出来的描述Tensor中元素数据类型的类</td>
		<td> tensor_name: 被描述的Tensor的name </td>
	</tr>
	<tr>
		<td><pre> const std::string& tensor_name() const</pre></td>
		<td> 获取tensor的name</td>
		<td> / </td>
	</tr>
	<tr>
		<td rowspan="1"> DrrPatternContext</td>
		<td><pre>drr::SourcePattern DrrPatternContext::SourcePattern()</pre> </td>
		<td> 创建一个SourcePattern对象，并返回 </td>
		<td> / </td>
	</tr>
</table>

## 3 使用示例
Example 1: Matmul + Add -> FusedGemmEpilogue
~~~ c++
class FusedLinearPattern : public pir::drr::DrrPatternBase<FusedLinearPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    // 声明Source Pattern
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul = pat.Op(paddle::dialect::MatmulOp::name(),
                                {{"transpose_x", pat.Attr("trans_x")},
                                 {"transpose_y", pat.Attr("trans_y")}});
    const auto &add = pat.Op(paddle::dialect::AddOp::name());

    pat.Tensor("tmp") = matmul(pat.Tensor("x"), pat.Tensor("w"));
    pat.Tensor("out") = add(pat.Tensor("tmp"), pat.Tensor("bias"));

    // 声明Result Pattern
    pir::drr::ResultPattern res = pat.ResultPattern();
    // 声明Constrain
    const auto &act_attr =
        res.Attr([](const pir::drr::MatchContext &match_ctx) -> std::any {
          return "none";
        });
    const auto &fused_gemm_epilogue = res.Op(paddle::dialect::FusedGemmEpilogueOp::name(),
                                             {{{"trans_x", pat.Attr("trans_x")},
                                               {"trans_y", pat.Attr("trans_y")},
                                               {"activation", act_attr}}});
    fused_gemm_epilogue(
        {&res.Tensor("x"), &res.Tensor("w"), &res.Tensor("bias")},
        {&res.Tensor("out")});
  }
};
~~~

Example 2: Full + Expand -> Full
~~~ c++
class FoldExpandToConstantPattern
    : public pir::drr::DrrPatternBase<FoldExpandToConstantPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    // 声明Source Pattern
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &full1 = pat.Op(paddle::dialect::FullOp::name(),
                               {{"shape", pat.Attr("shape_1")},
                                {"value", pat.Attr("value_1")},
                                {"dtype", pat.Attr("dtype_1")},
                                {"place", pat.Attr("place_1")}});
    const auto &full_int_array1 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"value", pat.Attr("expand_shape_value")},
                {"dtype", pat.Attr("dtype_2")},
                {"place", pat.Attr("place_2")}});
    const auto &expand = pat.Op(paddle::dialect::ExpandOp::name());
    pat.Tensor("ret") = expand(full1(), full_int_array1());

    // 声明Result Pattern      Constrains: 本Pass无额外约束规则
    pir::drr::ResultPattern res = pat.ResultPattern();
    const auto &full2 = res.Op(paddle::dialect::FullOp::name(),
                               {{"shape", pat.Attr("expand_shape_value")},
                                {"value", pat.Attr("value_1")},
                                {"dtype", pat.Attr("dtype_1")},
                                {"place", pat.Attr("place_1")}});
    res.Tensor("ret") = full2();
  }
};
~~~
