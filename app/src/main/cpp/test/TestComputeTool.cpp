//
// Created by 神经元002 on 2017/3/9.
//
#include <utils.h>
#include <ComputeTool.h>
extern "C" {

JNIEXPORT void JNICALL
Java_com_compilesense_liuyi_caffemodelinput_NativeTest_testComputeIm2Col(
        JNIEnv *env, jobject instance
) {
    float * input = new float[50]{
            1,2,3,4,5,
            1,2,3,4,5,
            1,2,3,4,5,
            1,2,3,4,5,
            1,2,3,4,5,
            1,2,3,4,5,
            1,2,3,4,5,
            1,2,3,4,5,
            1,2,3,4,5,
            1,2,3,4,5
    };
    std::vector<size_t> inputShape(3);
    inputShape[0] = 5;
    inputShape[1] = 5;
    inputShape[2] = 2;
    MultiDimData<float> inputData(input, inputShape);

    float * kernel = new float[9]{
            1,2,3,4,5,6,7,8,9
    };
    std::vector<size_t> kernelShape(2);
    kernelShape[0] = 3;
    kernelShape[1] = 3;
    MultiDimData<float> kernelData(kernel, kernelShape);

    ConvolutionLayerParams convolutionLayerParams;

    double  st = now_ms();
    MultiDimData<float> * outputPtr =  im2col(&inputData, &kernelData, convolutionLayerParams);
    LOGE("finish im2col, time: %f", now_ms() - st);
    LOGE("output dimens: %ui", (unsigned int) outputPtr->num_dimensions);
    int channel,height,width;
    channel = (int) outputPtr->get_c();
    height = (int) outputPtr->get_h();
    width = (int) outputPtr->get_w();
    for (int c = 0; c < channel; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int index = c * height * width + h * width + w;
                LOGE("output[%i][%i][%i]=%f",c,h,w,outputPtr->data_ptr[index]);
            }
        }
    }

    LOGE("release data");
}

}


