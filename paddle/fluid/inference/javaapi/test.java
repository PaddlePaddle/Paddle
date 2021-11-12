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
        //config.enableProfile();
        config.enableMKLDNN();
        try {
            System.out.println(config.getCppModelDir());
            System.out.println(config.getCppParamsFile());
            System.out.println(config.getProgFile());
        } catch (Exception e) {
            e.printStackTrace();
        }
        config.getCpuMathLibraryNumThreads();
        config.getFractionOfGpuMemoryForPool();
        //config.switchIrDebug(true);
        System.out.println(config.summary());

        Predictor predictor = Predictor.createPaddlePredictor(config);

        long n = predictor.getInputNum();

        String inNames = predictor.getInputNameById(0);

        Tensor inHandle = predictor.getInputHandle(inNames);

        inHandle.Reshape(4, new int[]{1, 3, 224, 224});

        float[] inData = new float[1*3*224*224];
        inHandle.CopyFromCpu(inData);
        predictor.run();
        String outNames = predictor.getOutputNameById(0);
        Tensor outHandle = predictor.getOutputHandle(outNames);
        float[] outData = new float[outHandle.GetSize()];
        outHandle.CopyToCpu(outData);

        predictor.TryShrinkMemory();
        predictor.ClearIntermediateTensor();

        System.out.println(outData[0]);
        System.out.println(outData.length);
    }
}
