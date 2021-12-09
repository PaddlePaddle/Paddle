#ifdef TS_OPS_BASE
#else
#define TS_OPS_BASE

// Tensor shape dialect.
def TensorShapeDialect : Dialect {
    let name = "ts";

    let description = [{
        The Tensor Shape dialect.

        This dialect contains operations for working with tensor shapes.
    }];

    let cppNamespace = "::infrt::ts";
}

// Type definition.
def TS_Shape : DialectType<TensorShapeDialect,
CPred<"$_self.isa<::infrt::ts::ShapeType>()">, "!ts.shape type">,
BuildableType<"$_builder.getType<::infrt::ts::ShapeType>()"> {
    let typeDescription = [{
        `!ts.shape type` represents a static tensor shape.
}];
}

def TS_PartialShape : DialectType<TensorShapeDialect,
CPred<"$_self.isa<::infrt::ts::PartialShapeType>()">, "!ts.partial_shape type">,
BuildableType<"$_builder.getType<::infrt::ts::PartialShapeType>()"> {
    let typeDescription = [{
        `!ts.partial_shape type` represents either a static tensor shape, unranked
        tensor shape or a ranked tensor shape with unknown dimension sizes.
}];
}

#endif  // TS_OPS_BASE
