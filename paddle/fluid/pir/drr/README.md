# DRR (Declarative Rewrite Rule) Tool User Manual
---
## 1. Related Background

PASS is a key component for optimizing IR, and the transformation of DAG-to-DAG is the most common pass type. The DAG to DAG type transformation refers to the process of replacing one DAG subgraph in the original graph with another DAG subgraph. This process can be divided into two steps: matching and rewriting. In the matching stage, it is necessary to fully match the original subgraph in the Program based on the organizational structure of Tensor and Op. In the rewriting stage, the original subgraph is replaced with the target subgraph, and the input and output of the target subgraph must be a subset of the input and output of the original graph.

In order to reduce the development cost of PASS and allow users to focus on processing optimization logic without worrying about the underlying IR data structure, we have developed a DRR (Declarative Rewrite Rule) tool based on declarative rewriting to handle this type of Pattern Rewrite PASS. Users can declare patterns between the original subgraph and the target subgraph through a simple and easy-to-use interface, and the DRR tool can automatically match the source pattern in the Program and replace it with the result pattern.

Constant folding refers to the fact that Ops containing constants in operands can usually be collapsed into result constant values. Constant folding is the most common degenerate version of DAG-to-DAG type transformations. For ease of understanding, here is a simple example of using the DRR interface to implement constant folding:
~~~ c++
// 1. First, inherit the specialization template class of DrPatternBase
class RemoveRedundentCastPattern
    : public pir::drr::DrrPatternBase<RemoveRedundentCastPattern> {
    // 2. Overwritten operator () overload function in this class
	void operator()(pir::drr::DrrPatternContext *ctx) const override {
		// 3. Declare a SourcePattern containing two consecutive castOps using Op, Tensor, and Attribute
	    auto pat = ctx->SourcePattern();
	    pat.Tensor("tmp") = pat.Op(
	        "pd_op.cast", {{"dtype", pat.Attr("dtype1")}})(pat.Tensor("arg0"));
	    pat.Tensor("ret") = pat.Op(
	        "pd_op.cast", {{"dtype", pat.Attr("dtype2")}})(pat.Tensor("tmp"));

		// 4. Declare a ResultPattern
	    auto res = pat.ResultPattern();
	    res.Tensor("ret") = res.Op(
	        "pd_op.cast", {{"dtype", pat.Attr("dtype2")}})(res.Tensor("arg0"));
  }
};
~~~

As can be seen from the example code, DRR is mainly composed of the following three major components:
+ `Source Pattern`：Used to describe the pattern subgraph to be matched in the Program
+ `Result Pattern`：Used to describe the pattern subgraph replaced with
+ `Constrains`：Used to specify restrictions for substitution
Developers need to specify the pattern subgraph to be matched through `Source Pattern`, specify constraints through `Constrains`, and specify the subgraph to replace with through `Result Pattern` to achieve a complete Pass. Compared to existing Pass development, which requires developers to be familiar with the underlying data structure, developers under the DRR Pass API only need to focus on the following DRR primitives：

<table>
	<tr>
		<th> Keywords </th>
		<th> Source Pattern </th>
		<th> Result Pattern </th>
		<th> Constrain </th>
	 </tr>
	 <tr>
		<td rowspan="3">Op</td>
		<td> <pre> source_pattern.Op("op_name", {{"attr_name", source_pattern.Attr("attr_var_name")}})</pre></td>
		<td rowspan="3"><pre> result_pattern.Op("op_name", {{"attr_name", result_pattern.Attr("attr_var_name")}})</pre></td>
		<td rowspan="3">Without an API, the OP call relationship is a constraint relationship, which has been determined in the SourcePattern </td>
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


**Note: DRR only supports matching and replacing the SourcePattern and ResultPattern of closures. If the declared subgraph is not a closure, unknown errors may occur**
## 2. Interface List

<table>
	 <tr>
		<th> Class </th>
		<th> Function </th>
		<th> Function Description </th>
		<th> Parameter Interpretation </th>
	 </tr>
	<tr>
		<td rowspan="2"> DrrPatternBase </td>
		<td><pre> virtual void operator()(pir::drr::DrrPatternContext* ctx) const </pre></td>
		<td> This class is the entry point for users to declare and rewrite SourcePattern and ResultPattern. Users need to customize a class A and inherit the specialized template class DrrPatternBase&lt;A&gt;. By implementing the reserved operator() interface in DrPatternBase, the declaration can be completed. </td>
		<td> ctx: The context to which the current Pattern belongs</td>
	</tr>
	<tr>
		<td><pre> std::unique_ptr&lt;DrrRewritePattern&gt; Build(
      pir::IrContext* ir_context, pir::PatternBenefit benefit = 1) const </pre></td>
		<td> Users can add user-defined patterns through this interface </td>
		<td> ir_context: The ir context in which the current pattern is located </td>
	</tr>
	<tr>
		<td rowspan="5"> SourcePattern</td>
		<td><pre> Attribute Attr(const std::string& attr_name) const </pre></td>
		<td> Declare an attr named in SourcePattern_ Properties of name </td>
		<td> attr_name: The name of the attribute must be unique within the SourcePattern</td>
	</tr>
	<tr>
		<td><pre> const drr::Tensor& Tensor(const std::string& tensor_name)</pre></td>
		<td> Declare a tensor named in SourcePattern_ Tensor of name</td>
		<td>  tensor_name: The name of the declared Tensor must be unique within the SourcePattern </td>
	</tr>
	<tr>
		<td><pre> const drr::Op& Op(const std::string& op_type, const std::unordered_map&lt;std::string, Attribute&gt;& attributes = {})</pre></td>
		<td> Declare an Op in the SourcePattern</td>
		<td> op_type: The declared op name can be obtained through the paddle::dialect::xxOp
	::name() interface or directly passed in as the name of the Op <br> attributes : Attribute information of the created op </td>
	</tr>
	<tr>
		<td><pre>void RequireEqual(const TensorShape& first, const TensorShape& second)</pre></td>
		<td> Declare that the TensorShapes of two Tensors in the SourcePattern are the same</td>
		<td> first: TensorShape of the first Tensor <br> second : TensorShape of the second Tensor</td>
	</tr>
	<tr>
		<td> <pre>void RequireNativeCall(const std::function&lt;bool(const MatchContext&)&gt;& custom_fn)</pre></td>
		<td> Declare a constraint in the SourcePattern, which users can use to implement custom constraints on the SourcePattern using this interface and lamda expressions</td>
		<td> custom_fn: User defined constraint function</td>
	</tr>
	<tr>
		<td rowspan="6">ResultPattern</td>
		<td><pre>Attribute Attr(const std::string& attr_name) const </pre></td>
		<td> Declare an attribute named attr_name in ResultPattern</td>
		<td> attr_name: The name of the attribute must be unique within the SourcePattern </td>
	</tr>
	<tr>
		<td> <pre>const drr::Tensor& Tensor(const std::string& tensor_name)</pre></td>
		<td> Declare a tensor named tensor_name in SourcePattern</td>
		<td> tensor_name: The name of the declared Tensor must be unique within the SourcePattern </td>
	</tr>
	<tr>
		<td><pre> const drr::Op& Op(
      const std::string& op_type,
      const std::unordered_map&lt;std::string, Attribute&gt;& attributes = {}) </pre></td>
		<td> Declare an Op in the SourcePattern </td>
		<td> op_type: The declared op name can be obtained through the paddle::dialect::xxOp
	::name() interface or directly passed in as the name of the Op <br> attributes:  Attribute information of the created op </td>
	</tr>
	<tr>
		<td><pre> void RequireEqual(const TensorShape& first, const TensorShape& second)</pre></td>
		<td> Declare that the TensorShapes of two Tensors in the SourcePattern are the same</td>
		<td> first: TensorShape of the first Tensor <br> second: TensorShape of the second Tensor </td>
	</tr>
	<tr>
		<td><pre>void RequireEqual(const TensorDataType& first, const TensorDataType& second)</pre></td>
		<td> Declare that the data types of two Tensors in SourcePattern are the same</td>
		<td> first: DataType of the first Tensor <br> second : DataType of the second Tensor</td>
	</tr>
		<tr>
		<td><pre> void RequireNativeCall(const std::function&lt;bool(const MatchContext&)&gt;& custom_fn)</pre></td>
		<td> Declare a constraint in SourcePattern, and users can use this interface and lamda expression to implement custom constraints on SourcePattern</td>
		<td> custom_fn: User defined constraint function </td>
	</tr>
	<tr>
		<td rowspan="2"> TensorShape</td>
		<td><pre>explicit TensorShape(const std::string& tensor_name) </pre></td>
		<td> The class abstracted to describe the shape of the Tensor </td>
		<td> tensor_name: Name of the described sensor </td>
	</tr>
	<tr>
		<td><pre> const std::string& tensor_name() const</pre></td>
		<td> Get the name of the Tensor</td>
		<td>  None </td>
	</tr>
	<tr>
		<td rowspan="2"> TensorDataType</td>
		<td><pre>explicit TensorDataType(const std::string& tensor_name)</pre></td>
		<td> Abstract class describing element data type in Tensor</td>
		<td> tensor_name: Name of the described sensor </td>
	</tr>
	<tr>
		<td> <pre>const std::string& tensor_name() const</pre></td>
		<td> Get the name of the supplier</td>
		<td> None </td>
	</tr>
	<tr>
		<td rowspan="1"> DrrPatternContext</td>
		<td><pre>drr::SourcePattern DrrPatternContext::SourcePattern() </pre></td>
		<td> Create a SourcePattern object and return </td>
		<td> None </td>
	</tr>
</table>

## 3 Example
Example 1: Matmul + Add -> FusedGemmEpilogue
~~~ c++
class FusedLinearPattern : public pir::drr::DrrPatternBase<FusedLinearPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
	// Declare SourcePattern
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul = pat.Op(paddle::dialect::MatmulOp::name(),
                                {{"transpose_x", pat.Attr("trans_x")},
                                 {"transpose_y", pat.Attr("trans_y")}});
    const auto &add = pat.Op(paddle::dialect::AddOp::name());

    pat.Tensor("tmp") = matmul(pat.Tensor("x"), pat.Tensor("w"));
    pat.Tensor("out") = add(pat.Tensor("tmp"), pat.Tensor("bias"));

    // Declare ResultPattern
    pir::drr::ResultPattern res = pat.ResultPattern();
    // Declare Constrain
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
    // Declare SourcePattern
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

    // Declare ResultPattern
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
