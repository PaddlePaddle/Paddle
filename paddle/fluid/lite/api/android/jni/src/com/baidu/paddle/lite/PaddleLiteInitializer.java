package com.baidu.paddle.lite;

/**
 * Initializer for PaddleLite.
 */
public class PaddleLiteInitializer {

    /** name of C++ JNI lib */
    public final static String JNI_LIB_NAME = "paddle_lite_jni";

    /** 
     * load the C++ JNI lib
     * @return true if initialize successfully.
     */
    public static boolean init() {
        System.loadLibrary(JNI_LIB_NAME);
        initNative();
        return true;
    }
    
    private static native void initNative();
    
    static {
        init();
    }
}
