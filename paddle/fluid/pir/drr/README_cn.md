# DRR( Declarative Rewrite Rule)工具用户使用手册
---
## 1. 相关背景
目前Paddle正在进行IR升级工作，由于PASS为对IR进行优化的关键组件，因此对于现有的所有PASS也需要基于新IR重新进行设计和实现。在经过统计分析发现，需要重写的Pass中属于DAG->DAG PatternRewrite类型的Pass数量占比过半。一方面为了提升用户在新IR上开发Pass的使用体验，另一方面为了降低全量Pass的迁移成本，我们开发了基于声明式重写的DRR ( Declarative Rewrite Rule ) 工具来处理这种PatternRewrite类型的Pass。用户可以通过一套简洁易用的接口对原有子图和目标子图进行模式声明，DRR工具就能自动的在Program中对原图进行匹配，并替换成目标子图。

![img1](https://github.com/gongshaotian/Paddle/assets/141618702/942a9f69-7e21-47bf-a479-933c551b2d92)

DRR(Declarative Rewrite Rule) Pass API并不是IR，而是对IR的统一封装，目的是让用户集中在对优化逻辑的处理上，而不需要关心对底层IR的处理。DRR主要由以下三大组件构成：
+ Source Pattern：用于描述在Program中待匹配的模式子图
+ Result Pattern：用于描述需要替换为的模式子图
+ Constrains：用于指定进行替换的限制条件

### 1.1 基于声明式接口构建DAG子图
![img2](https://github.com/gongshaotian/Paddle/assets/141618702/c28cced9-41dd-42ea-b3f0-b3fb4e5016a7)

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
pat.Op()会根据传入的name信息创建对应的Op对象，Op对象在传入Tensor对象执行时便会在Pattern Graph中创建对应的OpCall对象，同时在DAG中建立OpCall与Tensor的连接关系，这个阶段仅会根据用户代码创建中间DAG子图，此时并不会进行与新IR Program的匹配替换流程。
## 2. 接口文档
### 2.1 template \<typename DrrPattern\> Class DrrPatternBase
---
该类是用户进行 SourcePattern 和 ResultPattern 声明和重写的入口。用户需要自定义一个类 A，并且继承特化的模版类 DrrPatternBase\<A\>，然后再实现 DrrPatternBase 中预留的 operator() 接口即可完成声明。

#### 2.1.1 DrrPatternBase::operator()
~~~ c++
virtual void operator()(pir::drr::DrrPatternContext* ctx) const
~~~
用户通过在子类中重写该虚函数，实现对SourcePattern和ResultPattern的声明

Example：
~~~ c++
class ExampleAPattern
    : public pir::drr::DrrPatternBase<ExampleAPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    // SourcePattern 创建
	pir::drr::SourcePattern pat = ctx->SourcePattern();
	// SourcePattern 的模式声明
	...

    // ResultPattern 创建
    pir::drr::ResultPattern res = pat.ResultPattern();
    // ResultPattern 的模式创建
    ...
  }
};
~~~

#### 2.1.2 DrrPatternBase::Build()
~~~ c++
std::unique_ptr<DrrRewritePattern> Build(
      pir::IrContext* ir_context, pir::PatternBenefit benefit = 1) const
~~~
用户可以通过该接口实现用户自定义Pattern的添加，参数：
+ *ir_context :* 当前Pattern所在的ir上下文
+ *benefit :* 表示当前Pattern实现的Pass的所处的级别，具体分级可参考：
	+ *benefit = 0 :* 框架需要的最基本的Pass
	+ *benefit = 1 :* 常量折叠、cse、内存优化等
	+ *benefit = 2 :* 逻辑上对Op进行融合的Pass
	+ *benefit = 3:* layout等类型的Pass

Example：
~~~ c++
class DrrPatternRewritePass : public pir::Pass {
 public:
  DrrPatternRewritePass() : pir::Pass("DrrPatternRewritePass", 1) {}

  bool Initialize(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(ExampleAPattern().Build(context));
    ps.Add(ExampleBPattern().Build(context));
    ps.Add(ExampleCPattern().Build(context));
    ps.Add(ExampleDPattern().Build(context));
    ps.Add(ExampleEPattern().Build(context));

    patterns_ = pir::FrozenRewritePatternSet(std::move(ps));
    return true;
  }~~~

### 2.2 Class Source Pattern
---
#### 2.2.1 SourcePattern::Attr()
~~~ c++
Attribute Attr(const std::string& attr_name) const
~~~
在SourcePattern中声明一个名为attr_name的属性， 参数：
+ *attr_name :* 属性的名称，需要满足SourcePattern内唯一

#### 2.2.2 SourcePattern::Tensor()
~~~ c++
const drr::Tensor& Tensor(const std::string& tensor_name)
~~~
在SourcePattern中声明一个名为tensor_name的tensor，参数：
+ *tensor_name :* 声明的Tensor的名称，需要满足SourcePattern内唯一

#### 2.2.3 SourcePattern::Op()
~~~ c++
const drr::Op& Op(
      const std::string& op_type,
      const std::unordered_map<std::string, Attribute>& attributes = {})
~~~
在SourcePattern中声明一个Op，参数：
+ *op_type :* 声明的op名称，需要满足 "pd_op.xxx"的格式。例如 : "pd_op.reshape"
+ *attributes :* 所创建的op的属性信息

#### 2.2.4 SourcePattern::RequireEqual()
~~~ c++
void RequireEqual(const TensorShape& first, const TensorShape& second)
~~~
声明SourcePattern中两个Tensor的TensorShape相同，参数：
+ *first :*  第一个Tensor的TensorShape
+ *second :* 第二个Tensor的TensorShape

~~~ c++
void RequireEqual(const TensorDataType& first, const TensorDataType& second)
~~~
声明SourcePattern中两个Tensor的数据类型相同，参数:
+ *first :* 第一个Tensor的DataType
+ *second :* 第二个Tensor的DataType

#### 2.2.5 SourcePattern::RequireNativeCall()
~~~ c++
void RequireNativeCall(const std::function<bool(const MatchContext&)>& custom_fn)
~~~
在SourcePattern中声明一个Native约束，用户可以利用此接口和lamda表达式实现对SourcePattern的自定义约束

Example：
~~~ c++
void SimpliedRemoveRedundentCastPass(DrrPassContext* ctx) {
  SourcePattern pat = ctx->SourcePattern();
  const auto& Cast = pat.Op("cast");
  pat.Tensor("ret") = Cast(Cast(pat.Tensor("arg0"));
  
  pat.RequireNativeCall(
	[](MatchContext* match_ctx) -> bool 
	{
		return match_ctx->Tensor("ret").dtype<common::Type>() == 
			   match_ctx->Tensor("arg0").dtype<common::Type>();
	}
  );
  
  ResultPattern res = pat.ResultPattern();
  res.Tensor("ret").Assign(pat.Tensor("arg0"));
}
~~~

### 2.3 Class ResultPattern
---
#### 2.3.1 ResultPattern::Attr()
~~~ c++
Attribute Attr(const std::string& attr_name) const
~~~
在 ResultPattern 中声明一个名为 attr_name 的属性， 参数：
+ *attr_name :* 属性的名称，需要满足 ResultPattern 内唯一

~~~ c++
using AttrComputeFunc = std::function<std::any(const MatchContext&)>;

Attribute Attr(const AttrComputeFunc& attr_compute_func) const
~~~
在 ResultPattern 中声明属性，用户利用此接口和lamda表达式实现根据已有的属性到新属性的自定义生成规则

Example：
~~~ c++
class FusedLinearPattern : public pir::drr::DrrPatternBase<FusedLinearPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul = pat.Op("pd_op.matmul",
                                {{"transpose_x", pat.Attr("trans_x")},
                                 {"transpose_y", pat.Attr("trans_y")}});
    const auto &add = pat.Op("pd_op.add");

    pat.Tensor("tmp") = matmul(pat.Tensor("x"), pat.Tensor("w"));
    pat.Tensor("out") = add(pat.Tensor("tmp"), pat.Tensor("bias"));

    // Result patterns：要替换为的子图
    pir::drr::ResultPattern res = pat.ResultPattern();
    const auto &act_attr =
        res.Attr([](const pir::drr::MatchContext &match_ctx) -> std::any {
          return "none";
        });
    const auto &fused_gemm_epilogue = res.Op("pd_op.fused_gemm_epilogue",
                                             {{{"trans_x", pat.Attr("trans_x")},
                                               {"trans_y", pat.Attr("trans_y")},
                                               {"activation", act_attr}}});
    fused_gemm_epilogue(
        {&res.Tensor("x"), &res.Tensor("w"), &res.Tensor("bias")},
        {&res.Tensor("out")});
  }
};
~~~

#### 2.3.2 ResultPattern::Tensor()
~~~ c++
drr::Tensor& Tensor(const std::string& tensor_name)
~~~
在 ResultPattern中声明一个名为 tensor_name 的 tensor，参数：
+ *tensor_name :* 声明的Tensor的名称，需要满足 ResultPattern 内唯一

#### 2.3.3 ResultPattern::Op()
~~~ c++
const drr::Op& Op(
    const std::string& op_type,
    const std::unordered_map<std::string, Attribute>& attributes = {})
~~~
在 ResultPattern 中声明一个 Op，参数：
+ *op_type :* 声明的 op 名称，需要满足 "pd_op.xxx" 的格式。例如 : "pd_op.reshape"
+ *attributes :* 所创建的 op 的属性信息

#### 2.3.4 ResultPattern::NoneTensor()
~~~ c++
drr::Tensor& NoneTensor()
~~~
在 ResultPattern 中声明一个NoneTensor，NoneTensor 的主要功能是用来占位。

当一个 Op 的输入Tensor 是一个可选项时，可以使用 NoneTensor 来占位，Example:
~~~ c++

fused_gemm_epilogue_grad(
						 {&res.Tensor("x"),
		                  &res.Tensor("w"),
	                      &res.NoneTensor(),
	                      &res.Tensor("out_grad")
                         },
                         {&res.Tensor("x_grad"),
                          &res.Tensor("w_grad"),
                          &res.Tensor("bias_grad")
                         });     
~~~

当一个 Op 的输出 Tensor 是一个不会被使用的 Tensor 时，也可以使用 NoneTensor()来占位，Example：
~~~ c++
reshape_5({&res.Tensor("concat_1_out")},
		  {&res.Tensor("reshape_5_out"), &res.NoneTensor()});
~~~

### 2.4 Class TensorShape
---
~~~ c++
explicit TensorShape(const std::string& tensor_name)
~~~
抽象出来描述Tensor的shape的类，参数：
+ *tensor_name :* 被描述的Tensor的name

#### 2.4.1 TensorShape::tensor_name()
~~~ c++
const std::string& tensor_name() const
~~~
获取tensor的name

### 2.5 Class TensorDataType
---
~~~ c++
explicit TensorDataType(const std::string& tensor_name)
~~~
抽象出来的描述Tensor中元素数据类型的类，参数：
+ *tensor_name :* 被描述的Tensor的name

#### 2.5.1 TensorDataType::tensor_name()
~~~ c++
const std::string& tensor_name() const
~~~
获取tensor的name

### 2.6 Class DrrPatternContext
---
#### 2.6.1 DrrPatternContext::SourcePattern()
~~~ c++
drr::SourcePattern DrrPatternContext::SourcePattern()
~~~
创建一个SourcePattern对象，并返回

#### 2.6.2 DrrPatternContext::source_pattern_graph()
~~~ c++
std::shared_ptr<SourcePatternGraph> source_pattern_graph() const
~~~
返回PatternContext内部的SourcePatternGraph对象

#### 2.6.3 DrrPatternContext::constraints()
~~~ c++
std::vector<Constraint> constraints() const
~~~
返回PatternContext内部的约束列表 Constrains

#### 2.6.4 DrrPatternContext::result_pattern_graph()
~~~ c++
std::shared_ptr<ResultPatternGraph> result_pattern_graph() const
~~~
返回PatternContext内部的ResultPatternGraph对象


## 3 使用示例
Example 1: Matmul + Add -> FusedGemmEpilogue
~~~ c++
class FusedLinearPattern : public pir::drr::DrrPatternBase<FusedLinearPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul = pat.Op("pd_op.matmul",
                                {{"transpose_x", pat.Attr("trans_x")},
                                 {"transpose_y", pat.Attr("trans_y")}});
    const auto &add = pat.Op("pd_op.add");

    pat.Tensor("tmp") = matmul(pat.Tensor("x"), pat.Tensor("w"));
    pat.Tensor("out") = add(pat.Tensor("tmp"), pat.Tensor("bias"));

    // Result patterns：要替换为的子图
    pir::drr::ResultPattern res = pat.ResultPattern();
    const auto &act_attr =
        res.Attr([](const pir::drr::MatchContext &match_ctx) -> std::any {
          return "none";
        });
    const auto &fused_gemm_epilogue = res.Op("pd_op.fused_gemm_epilogue",
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
    // Source Pattern 中可匹配的类型包括 Op 和 Tensor
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &full1 = pat.Op("pd_op.full",
                               {{"shape", pat.Attr("shape_1")},
                                {"value", pat.Attr("value_1")},
                                {"dtype", pat.Attr("dtype_1")},
                                {"place", pat.Attr("place_1")}});
    const auto &full_int_array1 =
        pat.Op("pd_op.full_int_array",
               {{"value", pat.Attr("expand_shape_value")},
                {"dtype", pat.Attr("dtype_2")},
                {"place", pat.Attr("place_2")}});
    const auto &expand = pat.Op("pd_op.expand");
    pat.Tensor("ret") = expand(full1(), full_int_array1());

    // Result patterns：要替换为的子图.      Constrains: 本Pass无额外约束规则
    pir::drr::ResultPattern res = pat.ResultPattern();
    const auto &full2 = res.Op("pd_op.full",
                               {{"shape", pat.Attr("expand_shape_value")},
                                {"value", pat.Attr("value_1")},
                                {"dtype", pat.Attr("dtype_1")},
                                {"place", pat.Attr("place_1")}});
    res.Tensor("ret") = full2();
  }
};
~~~
