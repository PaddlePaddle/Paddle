package com.baidu.paddle.inference;

public class Predictor {

    private long cppPaddlePredictorPointer;

    private long inputNum;

    private long outputNum;

    public Predictor(Config config) {
        cppPaddlePredictorPointer = createPredictor(config.getCppPaddleConfigPointer());
        config.resetCppPaddleConfigPointer();
        inputNum = getInputNum(cppPaddlePredictorPointer);
        outputNum = getOutputNum(cppPaddlePredictorPointer);
    }

    public static Predictor createPaddlePredictor(Config config){
        Predictor predictor = new Predictor(config);
        return predictor;
    }

    public String getInputNameById(long id){
        return getInputNameByIndex(this.cppPaddlePredictorPointer, id);
    }

    public String getOutputNameById(long id){
        return getOutputNameByIndex(this.cppPaddlePredictorPointer, id);
    }

    public Tensor getInputHandle(String name){
        long tensorPointer = getInputHandleByName(this.cppPaddlePredictorPointer, name);
        return new Tensor(tensorPointer);
    }

    public Tensor getOutputHandle(String name){
        long tensorPointer = getOutputHandleByName(this.cppPaddlePredictorPointer, name);
        return new Tensor(tensorPointer);
    }

    public void ClearIntermediateTensor(){
        predictorClearIntermediateTensor(this.cppPaddlePredictorPointer);
    }

    public void TryShrinkMemory(){
        predictorTryShrinkMemory(this.cppPaddlePredictorPointer);
    }


    public boolean run(){
        return runPD(this.cppPaddlePredictorPointer);
    }

    public long getCppPaddlePredictorPointer() {
        return cppPaddlePredictorPointer;
    }

    public long getInputNum() {
        return inputNum;
    }

    public long getOutputNum() {
        return outputNum;
    }

    private native void predictorTryShrinkMemory(long cppPaddleConfigPointer);

    private native void predictorClearIntermediateTensor(long cppPaddleConfigPointer);

    private native long createPredictor(long cppPaddleConfigPointer);

    private native long getInputNum(long cppPaddlePredictorPointer);

    private native long getOutputNum(long cppPaddlePredictorPointer);

    private native String getInputNameByIndex(long cppPaddlePredictorPointer, long index);

    private native String getOutputNameByIndex(long cppPaddlePredictorPointer, long index);

    private native long getInputHandleByName(long cppPaddlePredictorPointer, String name);

    private native long getOutputHandleByName(long cppPaddlePredictorPointer, String name);

    private native boolean runPD(long cppPaddlePredictorPointer);
}
