# DRR (Declarative Rewrite Rule) Tool User Manual
---
## 1. Related Background

PASS is a key component for optimizing IR, and the transformation of DAG-to-DAG is the most common pass type. The implementation of PASS with DAG-to-DAG PatternRewrite type is mainly divided into two steps: matching and rewriting. In the matching stage, it is necessary to fully match the source pattern in the Program. In the rewriting stage, the original subgraph is replaced with the target subgraph, and the source pattern and result pattern must meet the requirement that the input and output of the two subgraphs are identical. In order to reduce the development cost of PASS, we have developed a DRR (Declarative Rewrite Rule) tool based on declarative rewriting to handle PASS of the PatternRewrite type. Users can declare source patterns and result patterns through a simple and easy-to-use interface, and DRR tools can automatically match source patterns in Program and replace them with result patterns.
![img1](https://github.com/gongshaotian/Paddle/assets/141618702/942a9f69-7e21-47bf-a479-933c551b2d92)
The DRR PASS API is not IR, but a unified encapsulation of IR, with the aim of allowing users to focus on optimizing logic processing without worrying about processing the underlying IR. DRR is mainly composed of the following three major components:
+ `Source Pattern`：Used to describe the pattern subgraph to be matched in the Program
+ `Result Pattern`：Used to describe the pattern subgraph replaced with
+ `Constrains`：Used to specify restrictions for substitution
Developers need to specify the pattern subgraph to be matched through `Source Pattern`, specify constraints through `Constrains`, and specify the subgraph to replace with through `Result Pattern` to achieve a complete Pass. Compared to existing Pass development, which requires developers to be familiar with the underlying data structure, developers under the DRR Pass API only need to focus on the following DRR primitives：

| Keywords      | Source Pattern                                                                                                                                                                                      | Result Pattern                                                                                                         | Constrain                                                                                                                                                 |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Op        | **source_pattern.Op**("op_name", {{"attr_name", source_pattern.Attr("attr_var_name")}})  **source_pattern.Op**({"op_name0", "op_name1"}) **source_pattern.Op**([](const string& op_name) -> bool {}) | Create op : **result_pattern.Op**("op_name", {{“attr_name", result_pattern.Attr("attr_var_name")}})                    | Without an API, the OP call relationship is a constraint relationship, which has been determined in the SourcePattern                                                                                               |
| Tensor    | **source_pattern.Tensor**("name")                                                                                                                                                                 | **result_pattern.Tensor**("name")                                                                                      | **srouce_pattern.RequireXXX**(pat.Tensor("name1").shape(), pat.Tensor("name2").shape())                                                                   |
| Attribute | **source_pattern.Attr**("attr_var_name")                                                                                                                                                            | **result_pattern.Attr**("attr_var_name")   **result_pattern.Attr**([](MatchContext* match_ctx) { return attr_value; }) | **srouce_pattern.RequireXXX**(pat.Attr("name1"), pat.Attr("name2"))      **srouce_pattern.RequireNativeCall**([](MatchContext* match_ctx) { return false;}) |

### 1.1 Building DAG Example Based on Declarative Interface
The execution and parsing process of the DRR API is somewhat similar to that of networking through the Python interface during compilation in static graph mode. Taking merging duplicate casts as an example, the code implemented using the DRR interface is as follows:
~~~ c++
class RemoveRedundentCastPattern
    : public pir::drr::DrrPatternBase<RemoveRedundentCastPattern> {
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
	// Declare SourcePattern
    auto pat = ctx->SourcePattern();
    pat.Tensor("tmp") = pat.Op(
        "pd_op.cast", {{"dtype", pat.Attr("dtype1")}})(pat.Tensor("arg0"));
    pat.Tensor("ret") = pat.Op(
        "pd_op.cast", {{"dtype", pat.Attr("dtype2")}})(pat.Tensor("tmp"));

	// Declare ResultPattern
    auto res = pat.ResultPattern();
    res.Tensor("ret") = res.Op(
        "pd_op.cast", {{"dtype", pat.Attr("dtype2")}})(res.Tensor("arg0"));
  }
};
~~~
In this simple example, we first inherit the specialization template class of DrrPatternBase, and then override the operator() overloaded function in this class. In the operator() function, we declared a SourcePattern containing two consecutive castOp using Op, Tensor, and Attribute. It is obvious that SourcePattern can be optimized for constant folding. We can use a castOp to achieve the desired effect that SourcePattern wants, that is, the ResultPattern we declare. It is obvious that SourcePattern can be optimized for constant folding. We can use a castOp to achieve the desired effect that SourcePattern wants, that is, the ResultPattern we declare.

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
		<td rowspan="2">template &lt;typename DrrPattern&gt; Class DrrPatternBase </td>
		<td> virtual void operator()(pir::drr::DrrPatternContext* ctx) const </td>
		<td> This class is the entry point for users to declare and rewrite SourcePattern and ResultPattern. Users need to customize a class A and inherit the specialized template class DrrPatternBase&lt;A&gt;. By implementing the reserved operator() interface in DrPatternBase, the declaration can be completed. </td>
		<td> ctx: The context to which the current Pattern belongs</td>
	</tr>
	<tr>
		<td> std::unique_ptr&lt;DrrRewritePattern&gt; Build(
      pir::IrContext* ir_context, pir::PatternBenefit benefit = 1) const </td>
		<td> Users can add user-defined patterns through this interface </td>
		<td> ir_context: The ir context in which the current pattern is located </td>
	</tr>
	<tr>
		<td rowspan="5">Class SourcePattern</td>
		<td> Attribute Attr(const std::string& attr_name) const </td>
		<td> Declare an attr named in SourcePattern_ Properties of name </td>
		<td> attr_name: The name of the attribute must be unique within the SourcePattern</td>
	</tr>
	<tr>
		<td> const drr::Tensor& Tensor(const std::string& tensor_name)</td>
		<td> Declare a tensor named in SourcePattern_ Tensor of name</td>
		<td>  tensor_name: The name of the declared Tensor must be unique within the SourcePattern </td>
	</tr>
	<tr>
		<td> const drr::Op& Op(const std::string& op_type, const std::unordered_map&lt;std::string, Attribute&gt;& attributes = {})</td>
		<td> Declare an Op in the SourcePattern</td>
		<td> op_type: The declared op name can be obtained through the paddle::dialect::xxOp
	::name() interface or directly passed in as the name of the Op <br> attributes : Attribute information of the created op </td>
	</tr>
	<tr>
		<td> void RequireEqual(const TensorShape& first, const TensorShape& second)</td>
		<td> Declare that the TensorShapes of two Tensors in the SourcePattern are the same</td>
		<td> first: TensorShape of the first Tensor <br> second : TensorShape of the second Tensor</td>
	</tr>
	<tr>
		<td> void RequireNativeCall(const std::function&lt;bool(const MatchContext&)&gt;& custom_fn)</td>
		<td> Declare a Native constraint in the SourcePattern, which users can use to implement custom constraints on the SourcePattern using this interface and lamda expressions</td>
		<td> custom_fn: User defined Native constraint function</td>
	</tr>
	<tr>
		<td rowspan="6">Class ResultPattern</td>
		<td>Attribute Attr(const std::string& attr_name) const </td>
		<td> Declare an attribute named attr_name in ResultPattern</td>
		<td> attr_name: The name of the attribute must be unique within the SourcePattern </td>
	</tr>
	<tr>
		<td> const drr::Tensor& Tensor(const std::string& tensor_name)</td>
		<td> Declare a tensor named tensor_name in SourcePattern</td>
		<td> tensor_name: The name of the declared Tensor must be unique within the SourcePattern </td>
	</tr>
	<tr>
		<td> const drr::Op& Op(
      const std::string& op_type,
      const std::unordered_map&lt;std::string, Attribute&gt;& attributes = {}) </td>
		<td> Declare an Op in the SourcePattern </td>
		<td> op_type: The declared op name can be obtained through the paddle::dialect::xxOp
	::name() interface or directly passed in as the name of the Op <br> attributes:  Attribute information of the created op </td>
	</tr>
	<tr>
		<td> void RequireEqual(const TensorShape& first, const TensorShape& second)</td>
		<td> Declare that the TensorShapes of two Tensors in the SourcePattern are the same</td>
		<td> first: TensorShape of the first Tensor <br> second: TensorShape of the second Tensor </td>
	</tr>
	<tr>
		<td> void RequireEqual(const TensorDataType& first, const TensorDataType& second)</td>
		<td> Declare that the data types of two Tensors in SourcePattern are the same</td>
		<td> first: DataType of the first Tensor <br> second : DataType of the second Tensor</td>
	</tr>
		<tr>
		<td> void RequireNativeCall(const std::function&lt;bool(const MatchContext&)&gt;& custom_fn)</td>
		<td> Declare a Native constraint in SourcePattern, and users can use this interface and lamda expression to implement custom constraints on SourcePattern</td>
		<td> custom_fn: User defined Native constraint function </td>
	</tr>
	<tr>
		<td rowspan="2">Class TensorShape</td>
		<td>explicit TensorShape(const std::string& tensor_name) </td>
		<td> The class abstracted to describe the shape of the Tensor </td>
		<td> tensor_name: Name of the described sensor </td>
	</tr>
	<tr>
		<td> const std::string& tensor_name() const</td>
		<td> Get the name of the Tensor</td>
		<td>  None </td>
	</tr>
	<tr>
		<td rowspan="2">Class TensorDataType</td>
		<td>explicit TensorDataType(const std::string& tensor_name)</td>
		<td> Abstract class describing element data type in Tensor</td>
		<td> tensor_name: Name of the described sensor </td>
	</tr>
	<tr>
		<td> const std::string& tensor_name() const</td>
		<td> Get the name of the supplier</td>
		<td> None </td>
	</tr>
	<tr>
		<td rowspan="4">Class DrrPatternContext</td>
		<td>drr::SourcePattern DrrPatternContext::SourcePattern() </td>
		<td> Create a SourcePattern object and return </td>
		<td> None </td>
	</tr>
	<tr>
		<td> std::shared_ptr&lt;SourcePatternGraph&gt; source_pattern_graph() const</td>
		<td> Return the SourcePatternGraph object inside the PatternContext </td>
		<td> None </td>
	</tr>
	<tr>
		<td> std::vector&lt;Constraint&gt; constraints() const</td>
		<td> Return the list of constraints within the PatternContext</td>
		<td> 无 </td>
	</tr>
	<tr>
		<td> std::shared_ptr&lt;ResultPatternGraph&gt; result_pattern_graph() const</td>
		<td> Return the ResultPatternGraph object inside the PatternContext</td>
		<td> 无 </td>
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
