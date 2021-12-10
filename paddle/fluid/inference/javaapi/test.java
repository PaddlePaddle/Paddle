import com.baidu.paddle.inference.Predictor;
import com.baidu.paddle.inference.Config;
import com.baidu.paddle.inference.Tensor;

public class test {

    static {
        System.loadLibrary("paddle_inference");
    }

    public static void main(String[] args) {
        Config config = new Config();

        config.setCppModel(args[0], args[1]);
        config.enableMemoryOptim(true);
        config.enableProfile();
        config.enableMKLDNN();

        System.out.println("summary:\n" + config.summary());
        System.out.println("model dir:\n" + config.getCppModelDir());
        System.out.println("prog file:\n" + config.getProgFile());
        System.out.println("params file:\n" + config.getCppParamsFile());

        config.getCpuMathLibraryNumThreads();
        config.getFractionOfGpuMemoryForPool();
        config.switchIrDebug(false);
        System.out.println(config.summary());

        Predictor predictor = Predictor.createPaddlePredictor(config);

        long n = predictor.getInputNum();

        String inNames = predictor.getInputNameById(0);

        Tensor inHandle = predictor.getInputHandle(inNames);

        inHandle.reshape(4, new int[]{1, 3, 224, 224});

        float[] inData = new float[1*3*224*224];
        inHandle.copyFromCpu(inData);
        predictor.run();
        String outNames = predictor.getOutputNameById(0);
        Tensor outHandle = predictor.getOutputHandle(outNames);
        float[] outData = new float[outHandle.getSize()];
        outHandle.copyToCpu(outData);

        predictor.tryShrinkMemory();
        predictor.clearIntermediateTensor();

        System.out.println(outData[0]);
        System.out.println(outData.length);

        outHandle.destroyNativeTensor();
        inHandle.destroyNativeTensor();
        predictor.destroyNativePredictor();

        Config newConfig = new Config();
        newConfig.setCppModelDir("/model_dir");
        newConfig.setCppProgFile("/prog_file");
        newConfig.setCppParamsFile("/param");
        System.out.println("model dir:\n" + newConfig.getCppModelDir());
        System.out.println("prog file:\n" + newConfig.getProgFile());
        System.out.println("params file:\n" + newConfig.getCppParamsFile());
        config.destroyNativeConfig();

    }
}
