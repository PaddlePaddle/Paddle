package com.baidu.paddle.lite;

public enum PrecisionType {
    UNKNOWN(0), FLOAT(1), INT8(2), INT32(3), ANY(4);

    public final int value;

    private PrecisionType(int value) {
        this.value = value;
    }
}
