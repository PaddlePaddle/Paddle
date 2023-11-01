# DRR( Declarative Rewrite Rule)工具用户使用手册
---
## 1. 相关背景

PASS是对IR进行优化的关键组件，而DAG-to-DAG的变换是最常见的Pass类型。DAG-to-DAG PatternRewrite类型的PASS的实现主要分为两步：匹配和重写。匹配阶段需要根据Tensor和Op的组织结构在Program中完全匹配到原有子图，在重写阶段将原有子图替换为目标子图，并且原有子图和目标子图需要满足两个子图的输入和输出完全相同。为了降低PASS的开发成本，我们开发了基于声明式重写的DRR ( Declarative Rewrite Rule ) 工具来处理这种PatternRewrite类型的PASS。用户可以通过一套简洁易用的接口对原有子图和目标子图进行模式声明，DRR工具就能自动的在Program中对原图进行匹配，并替换成目标子图。

DRR(Declarative Rewrite Rule) PASS API并不是IR，而是对IR的统一封装，目的是让用户集中在对优化逻辑的处理上，而不需要关心对底层IR的处理。DRR主要由以下三大组件构成：
+ `Source Pattern`：用于描述在Program中待匹配的模式子图
+ `Result Pattern`：用于描述需要替换为的模式子图
+ `Constrains`：用于指定进行替换的限制条件
开发者需通过`Source Pattern`指定待匹配的模式子图、通过`Constrains`指定限制条件、通过`Result Pattern`指定要替换为的子图即可实现一个完整的Pass。相比现有Pass开发需要开发者熟悉底层数据结构，DRR Pass API下开发者只需关注如下DRR原语：

<table>
	<tr>
		<th> 句法 </th>
		<th> Source Pattern </th>
		<th> Result Pattern </th>
		<th> Constrain </th>
	 </tr>
	 <tr>
		<td rowspan="3">Op</td>
		<td> <pre> source_pattern.Op("op_name", {{"attr_name", source_pattern.Attr("attr_var_name")}})</pre></td>
		<td rowspan="3"><pre> result_pattern.Op("op_name", {{"attr_name", result_pattern.Attr("attr_var_name")}})</pre></td>
		<td rowspan="3">无API，OP调用关系即为约束关系，已在SourcePattern中确定 </td>
	</tr>
	<tr>
		<td><pre> source_pattern.Op({"op_name0", "op_name1"}) </pre></td>
	</tr>
	<tr>
		<td><pre> source_pattern.Op([](const string& op_name) -> bool {}) </pre></td>
	</tr>
	<tr>
		<td> Tensor </td>
		<td><pre> source_pattern.Tensor("name") </pre></td>
		<td> <pre>result_pattern.Tensor("name") </pre></td>
		<td> <pre>srouce_pattern.RequireXXX(pat.Tensor("name1").shape(), pat.Tensor("name2").shape()) </pre></td>
	</tr>
	<tr>
		<td rowspan="2"> Attribute </td>
		<td rowspan="2"><pre> source_pattern.Attr("attr_var_name")</pre></td>
		<td><pre> result_pattern.Attr("attr_var_name")</pre></td>
		<td><pre>srouce_pattern.RequireXXX(pat.Attr("name1"), pat.Attr("name2"))</pre></td>
	</tr>
	<tr>
		<td><pre> result_pattern.Attr([](MatchContext* match_ctx) { return attr_value; })</pre></td>
		<td><pre> srouce_pattern.RequireNativeCall([](MatchContext* match_ctx) { return false;})</pre></td>
	</tr>
</table>

### 1.1 基于声明式接口构建中间DAG子图示例
DRR API的执行解析过程与静态图模式下编译期通过Python接口组网有些类似。以合并重复cast为例，使用DRR接口实现的代码如下所示：
~~~ c++
class RemoveRedundentCastPattern
    : public pir::drr::DrrPatternBase<RemoveRedundentCastPattern> {
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
	// 声明 SourcePattern
    auto pat = ctx->SourcePattern();
    pat.Tensor("tmp") = pat.Op(
        "pd_op.cast", {{"dtype", pat.Attr("dtype1")}})(pat.Tensor("arg0"));
    pat.Tensor("ret") = pat.Op(
        "pd_op.cast", {{"dtype", pat.Attr("dtype2")}})(pat.Tensor("tmp"));

	// 声明 ResultPattern
    auto res = pat.ResultPattern();
    res.Tensor("ret") = res.Op(
        "pd_op.cast", {{"dtype", pat.Attr("dtype2")}})(res.Tensor("arg0"));
  }
};
~~~
在这个简单的例子中，我们首先继承了DrrPatternBase的特化模版类，然后在这个类中重写了operator()重载函数。在operator() 函数中我们使用Op、Tensor和Attribute声明出了包含两个连续castOp的SourcePattern。很明显SourcePattern是可以进行常量折叠优化的，我们使用一个castOp就能达到SourcePattern想要的达到的效果，即我们声明出的ResultPattern。做完这些我们就完成了一个DAG-to-DAG PatternRewrite类型的PASS的声明。

**注意：DRR仅支持对闭包的 SourcePattern 和 ResultPattern 进行匹配替换，若声明出的子图不闭包可能会出现未知的错误**
## 2. 接口列表

<table>
	 <tr>
		<th> 类 </th>
		<th> 函数 </th>
		<th> 功能描述 </th>
		<th> 参数解释 </th>
	 </tr>
	<tr>
		<td rowspan="2">template &lt;typename DrrPattern&gt; Class DrrPatternBase</td>
		<td> <pre> virtual void operator()(pir::drr::DrrPatternContext* ctx) const </pre></td>
		<td> 该类是用户进行SourcePattern和ResultPattern声明和重写的入口。用户需要自定义一个类A，并且继承特化的模版类 DrrPatternBase&lt;A&gt;，然后再实现DrrPatternBase中预留的operator()接口即可完成声明 </td>
		<td> ctx: 当前Pattern所属的上下文</td>
	</tr>
	<tr>
		<td> <pre>std::unique_ptr&lt;DrrRewritePattern&gt; Build(
      pir::IrContext* ir_context, pir::PatternBenefit benefit = 1) const </pre></td>
		<td> 用户可以通过该接口实现用户自定义Pattern的添加 </td>
		<td> ir_context: 当前Pattern所在的ir上下文 </td>
	</tr>
	<tr>
		<td rowspan="5">Class SourcePattern</td>
		<td> <pre> Attribute Attr(const std::string& attr_name) const </pre></td>
		<td> 在SourcePattern中声明一个名为attr_name的属性 </td>
		<td> attr_name: 属性的名称，需要满足SourcePattern内唯一 </td>
	</tr>
	<tr>
		<td><pre> const drr::Tensor& Tensor(const std::string& tensor_name) </pre></td>
		<td> 在SourcePattern中声明一个名为tensor_name的tensor</td>
		<td>  tensor_name: 声明的Tensor的名称，需要满足SourcePattern内唯一 </td>
	</tr>
	<tr>
		<td><pre> const drr::Op& Op(const std::string& op_type, const std::unordered_map&lt;std::string, Attribute&gt;& attributes = {})</pre></td>
		<td> 在SourcePattern中声明一个Op</td>
		<td> op_type: 声明的op名称，可以通过paddle::dialect::xxOp
	::name()接口获取，或直接传入Op的名称 <br> attributes : 所创建的op的属性信息 </td>
	</tr>
	<tr>
		<td><pre> void RequireEqual(const TensorShape& first, const TensorShape& second)</pre></td>
		<td> 声明SourcePattern中两个Tensor的TensorShape相同</td>
		<td> first: 第一个Tensor的TensorShape <br> second : 第二个Tensor的TensorShape</td>
	</tr>
	<tr>
		<td> <pre>void RequireNativeCall(const std::function&lt;bool(const MatchContext&)&gt;& custom_fn)</pre></td>
		<td> 在SourcePattern中声明一个Native约束，用户可以利用此接口和lamda表达式实现对SourcePattern的自定义约束</td>
		<td> custom_fn: 用户自定义的Native约束函数</td>
	</tr>
	<tr>
		<td rowspan="6">Class ResultPattern</td>
		<td><pre>Attribute Attr(const std::string& attr_name) const </pre></td>
		<td> 在 ResultPattern 中声明一个名为 attr_name 的属性 </td>
		<td> attr_name: 属性的名称，需要满足SourcePattern内唯一 </td>
	</tr>
	<tr>
		<td> <pre>const drr::Tensor& Tensor(const std::string& tensor_name)</pre></td>
		<td> 在SourcePattern中声明一个名为tensor_name的tensor</td>
		<td> tensor_name: 声明的Tensor的名称，需要满足SourcePattern内唯一 </td>
	</tr>
	<tr>
		<td><pre> const drr::Op& Op(
      const std::string& op_type,
      const std::unordered_map&lt;std::string, Attribute&gt;& attributes = {}) </pre></td>
		<td> 在SourcePattern中声明一个Op </td>
		<td> op_type: 声明的op名称，可以通过paddle::dialect::xxOp
	::name()接口获取，或直接传入Op的名称<br> attributes : 所创建的op的属性信息 </td>
	</tr>
	<tr>
		<td><pre> void RequireEqual(const TensorShape& first, const TensorShape& second)</pre></td>
		<td> 声明SourcePattern中两个Tensor的TensorShape相同</td>
		<td> first:  第一个Tensor的TensorShape <br> second : 第二个Tensor的TensorShape </td>
	</tr>
	<tr>
		<td><pre> void RequireEqual(const TensorDataType& first, const TensorDataType& second)</pre></td>
		<td> 声明SourcePattern中两个Tensor的数据类型相同</td>
		<td> first: 第一个Tensor的DataType <br> second : 第二个Tensor的DataType</td>
	</tr>
		<tr>
		<td> <pre>void RequireNativeCall(const std::function&lt;bool(const MatchContext&)&gt;& custom_fn)</pre></td>
		<td> 在SourcePattern中声明一个Native约束，用户可以利用此接口和lamda表达式实现对SourcePattern的自定义约束</td>
		<td> custom_fn: 用户自定义的Native约束函数 </td>
	</tr>
	<tr>
		<td rowspan="2">Class TensorShape</td>
		<td><pre>explicit TensorShape(const std::string& tensor_name) </pre></td>
		<td> 抽象出来描述Tensor的shape的类 </td>
		<td> tensor_name: 被描述的Tensor的name </td>
	</tr>
	<tr>
		<td><pre> const std::string& tensor_name() const</pre></td>
		<td> 获取tensor的name</td>
		<td>  无 </td>
	</tr>
	<tr>
		<td rowspan="2">Class TensorDataType</td>
		<td><pre>explicit TensorDataType(const std::string& tensor_name)</pre></td>
		<td> 抽象出来的描述Tensor中元素数据类型的类</td>
		<td> tensor_name: 被描述的Tensor的name </td>
	</tr>
	<tr>
		<td><pre> const std::string& tensor_name() const</pre></td>
		<td> 获取tensor的name</td>
		<td> 无 </td>
	</tr>
	<tr>
		<td rowspan="4">Class DrrPatternContext</td>
		<td><pre>drr::SourcePattern DrrPatternContext::SourcePattern()</pre> </td>
		<td> 创建一个SourcePattern对象，并返回 </td>
		<td> 无 </td>
	</tr>
	<tr>
		<td> <pre>std::shared_ptr&lt;SourcePatternGraph&gt; source_pattern_graph() const</pre></td>
		<td> 返回PatternContext内部的SourcePatternGraph对象</td>
		<td> 无 </td>
	</tr>
	<tr>
		<td><pre>std::vector&lt;Constraint&gt; constraints() const</pre></td>
		<td> 返回PatternContext内部的约束列表 Constrains</td>
		<td> 无 </td>
	</tr>
	<tr>
		<td><pre>std::shared_ptr&lt;ResultPatternGraph&gt; result_pattern_graph() const</pre></td>
		<td> 返回PatternContext内部的ResultPatternGraph对象</td>
		<td> 无 </td>
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
