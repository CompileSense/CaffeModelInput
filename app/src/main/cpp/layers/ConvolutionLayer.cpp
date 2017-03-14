//
// Created by 神经元002 on 2017/3/3.
//

#include <string>
#include <jni.h>
#include "utils.h"
#include "ConvolutionLayer.h"
#include <math.h>
//#include <nnpack.h>

//bool initNnpack();
//void releaseNnpack();
//int convNnpack(
//        float * output, float * input, float * kernel, float * bias,
//        size_t batch_size,size_t input_channels, size_t output_channels,
//        unsigned int pad, unsigned int inputSize_h, unsigned int inputSize_w,
//        unsigned int kernelSize_h, unsigned int kernelSize_w
//);

ConvolutionLayer::ConvolutionLayer(const std::string &name, unsigned int stride, unsigned int pad, int group, bool nonLinear)
        : name(name), nonLinear(nonLinear), paramHadLoad(false){
    params.setParams(pad, stride, 1, 1);

    this->weightSize = 0;
    this->biasSize=0;
    this->weight=NULL;
    this->bias=NULL;
    this->weightShape=NULL;
}
ConvolutionLayer::ConvolutionLayer(const std::string &name, ConvolutionLayer::Params params,bool nonLinear)
        : name(name), params(params), nonLinear(nonLinear),paramHadLoad(false){
    this->weightSize = 0;
    this->biasSize=0;
    this->weight=NULL;
    this->bias=NULL;
    this->weightShape=NULL;
}

ConvolutionLayer::~ConvolutionLayer() {
    LOGD("ConvolutionLayer delete!");
}

void ConvolutionLayer::setKernel(float *weight, int weightSize,
                                 int *weightShape, int weightShapeLength,
                                 float *bias, int biasSize){
    this->weight = weight;
    this->bias = bias;
    this->weightSize = weightSize;
    this->biasSize = biasSize;
    this->weightShape = weightShape;
    this->weightShapeLength = weightShapeLength;

    this->paramHadLoad = true;
//    LOGD("ConvolutionLayer setKernel! name: %s", this->name.data());
//    LOGD("ConvolutionLayer setKernel! weightSize: %i , last data:%f", weightSize, weight[weightSize - 1]);
//    LOGD("ConvolutionLayer setKernel! biasSize: %i , last data:%f", biasSize, bias[biasSize - 1]);
}

void ConvolutionLayer::releaseKernel() {

//    LOGD("ConvolutionLayer releaseKernel! name: %s", this->name.data());
//    LOGD("ConvolutionLayer setKernel! paramHadLoad:%i",paramHadLoad);
//    LOGD("ConvolutionLayer setKernel! weightSize: %i , last data:%f, prt: %p", weightSize, weight[weightSize - 1], weight);
//    LOGD("ConvolutionLayer setKernel! biasSize: %i , last data:%f, prt: %p", biasSize, bias[biasSize -1], bias);
//    LOGD("ConvolutionLayer setKernel! weightShape: %i , last data:%i, prt: %p", weightShapeLength, weightShape[weightShapeLength -1], weightShape);
    if (this->paramHadLoad){
        try {
            if (this->weight != NULL){
                delete[] weight;
            }
            if (this->weightShape != NULL){
                delete[] weightShape;
            }
            if (this->bias != NULL){
                delete[] bias;
            }
        }catch (std::exception exception){
            LOGE("releaseKernel failed! file:%s, line:%i, info:%s", __FILE__, __LINE__, exception.what() );
        }
    }
    this->paramHadLoad = false;
}

/**
 * group 为1,
 */
void ConvolutionLayer::compute(float * input, std::vector<unsigned  int> inputShape, float * output){

    unsigned int pad_ = (unsigned int) params.pad;
    int stride_h = params.stride_h;
    int stride_w = params.stride_w;

    //input
    unsigned int n_i = inputShape[0];//数量
    unsigned int c_i = inputShape[1];//通道
    unsigned int h_i = inputShape[2];//高度
    unsigned int w_i = inputShape[3];//高度

    //kernel
    unsigned int n_k = (unsigned int) weightShape[0];//数量
    unsigned int c_k = (unsigned int) weightShape[1];//通道
    unsigned int h_k = (unsigned int) weightShape[2];//高度
    unsigned int w_k = (unsigned int) weightShape[3];//宽度

    //output
    int n_o = n_i;
    int h_o = (int) (ceil((h_i + 2 * pad_ - h_k) / ((float) (stride_h))) + 1);
    int w_o = (int) (ceil((w_i + 2 * pad_ - w_k) / ((float) (stride_w))) + 1);
    int c_o = n_k;

    //利用 stride 对 input 进行处理。
    if (stride_h > 1){
        h_i = h_i/stride_h;
    }
    if (stride_w > 1){
        w_i = w_i/stride_w;
    }

//    if (!initNnpack()){
//        return;
//    }
//
//    convNnpack(output, input, weight, bias,
//                n_i, c_i, c_o, pad_, h_i, w_i, h_k, w_k
//    );
//
//    releaseNnpack();
}


void computeOutputShape(int* inputShape, int* kernelShape, int* strideData, int* padData, int* outputShape){
    for (int i = 0; i < 2; ++i) {
        const int input_dim = inputShape[2+i];//四维 [n] [c] [h] [w]
//        const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
        const int output_dim =
                (input_dim + 2 * padData[i] - kernelShape[2+i]) / strideData[i] + 1;
//        outputShape
    }
}

//bool initNnpack(){
//    //初始化 nnpack;
//    nnp_status s = nnp_initialize();
//    if (s != nnp_status_success){
//        LOGD("nnpack init failed:%i", s);
//        return false;
//    } else{
//        return true;
//    }
//}
//
//void releaseNnpack(){
//    nnp_status s = nnp_deinitialize();
//    if (s != nnp_status_success){
//        LOGE("nnpack release failed:%i", s);
//    }
//}
//
///**
// * 使用nnpack来计算卷积
// * @param[out] output: 输出数据 4D。
// * @param[in]  input: 输入数据 4D。
// * @param[in]  kernel: weight 数据 4D。
// * @param[in]  bias: bias 数据 1D。
// */
//int convNnpack(
//        float * output, float * input, float * kernel, float * bias,
//        size_t batch_size, size_t input_channels, size_t output_channels,
//        unsigned int pad, unsigned int inputSize_h, unsigned int inputSize_w,
//        unsigned int kernelSize_h, unsigned int kernelSize_w
//){
//    enum nnp_convolution_algorithm algorithm = nnp_convolution_algorithm_auto;//卷积算法
//
////    batch_size = 1;//TODO 弄清作用
//    const struct nnp_padding input_padding = {pad, pad, pad, pad};
//    const struct nnp_size input_size ={ inputSize_h, inputSize_w };
//    const struct nnp_size kernel_size = { kernelSize_h, kernelSize_w };
//    const struct nnp_size output_size = {
//            .width = (input_padding.left + input_size.width + input_padding.right - kernel_size.width) + 1,
//            .height = (input_padding.top + input_size.height + input_padding.bottom - kernel_size.height) + 1
//    };
//
////    if (output ==NULL){
////        output = (float*)malloc(batch_size* output_channels * output_size.height * output_size.width * sizeof(float));
////    }
//
//    pthreadpool_t threadpool=NULL; //pthreadpool_create(4);
//
////    LOGD("intput: ****************************");
////    for (int i = 0; i < batch_size; i++){
////        for (int j = 0; j < input_channels; j++){
////            for (int k = 0; i < inputSize_h; i++){
////
////            }
////        }
////    }
////
////    LOGD("intput: ****************************");
//    nnp_convolution_output(
//            algorithm,
//            batch_size, input_channels, output_channels,
//            input_size, input_padding, kernel_size,
//            input, kernel, bias, output,
//            nnp_activation_relu,//f(x) = max(x,0)
//            NULL,
//            threadpool,
////            &computation_profile
//            NULL);
//    return 0;
//}

#ifdef __cplusplus
extern "C" {
#endif

//构造一个 conv layer 实例
JNIEXPORT jlong JNICALL
Java_com_compilesense_liuyi_caffemodelinput_caffecnn_ConvolutionLayer_createConvolutionLayer(
        JNIEnv *env, jobject instance,
        jstring name_,
        jintArray stride_,
        jintArray pad_,
        jint group_,
        jboolean nonLinear_
){
    //name
    const char * name = env->GetStringUTFChars(name_,NULL);
    //stride
    int strideSize = 0;
    int * stride = jIntArray2prt(env, stride_, &strideSize);
    //pad
    int padSize = 0;
    int * pad = jIntArray2prt(env, pad_, &padSize);
    //group
    int group = group_;
    //nonLinear
    bool nonLinear = nonLinear_;
    ConvolutionLayer * convLayerPrt =  new ConvolutionLayer(
            name,
            (unsigned int)stride[0],
            (unsigned int)pad[0],
            group,
            nonLinear
    );

    env->ReleaseStringUTFChars(name_,name);
    return (jlong) convLayerPrt;
}

//析构一个实例
JNIEXPORT void JNICALL
Java_com_compilesense_liuyi_caffemodelinput_caffecnn_ConvolutionLayer_deleteConvolutionLayer(
        JNIEnv *env, jobject instance,
        jlong objectPrt_
){
    ConvolutionLayer *objectPrt = (ConvolutionLayer *) objectPrt_;
    objectPrt->releaseKernel();
    delete(objectPrt);
}


JNIEXPORT void JNICALL
Java_com_compilesense_liuyi_caffemodelinput_caffecnn_ConvolutionLayer_setParam(
        JNIEnv *env, jobject instance,
        jlong objectPrt_,
        jobjectArray weight_,
        jintArray weightShape_,
        jfloatArray bias_
){
    ConvolutionLayer *objectPrt = (ConvolutionLayer *) objectPrt_;

    int weightShapeLength = 0;
    int* weightShape = jIntArray2prt(env, weightShape_, &weightShapeLength);


    int weightArrayLength = 0;
    float * weight = jFloatArrayFrom4DimensionJavaArray(
            env,
            weight_,
            weightShape,
            weightShapeLength
    );

    int biasArrayLength = 0;
    float* bias = jFloatArray2prt(env, bias_, &biasArrayLength);

//    objectPrt -> setKernel(weight_, weight, weightArrayLength,
//                          weightShape, weightArrayLength,
//                          bias_, bias, biasArrayLength);
}

//设置参数
JNIEXPORT void JNICALL
Java_com_compilesense_liuyi_caffemodelinput_caffecnn_ConvolutionLayer_setParam1DimenWeight(
        JNIEnv *env, jobject instance,
        jlong objectPrt_,
        jfloatArray weight_,
        jintArray weightShape_,//因为 int 是用 GetIntArrayRegion, 来访问的不需要专门释放 JVM。
        jfloatArray bias_
){
    ConvolutionLayer *objectPrt = (ConvolutionLayer *) objectPrt_;

    int weightShapeLength = 0;
    int* weightShape = jIntArray2prt(env, weightShape_, &weightShapeLength);

    int weightArrayLength = 0;
    float * weight = jFloatArray2prtFast(env, weight_, &weightArrayLength);

    int biasArrayLength = 0;
    float* bias = jFloatArray2prtFast(env, bias_, &biasArrayLength);

    objectPrt->setKernel(weight, weightArrayLength,
                         weightShape, weightShapeLength,
                         bias, biasArrayLength);

    jFloatArrayRelease(env, weight_, weight);
    jFloatArrayRelease(env, bias_, bias);
}

//计算测试 输入数据不应该是每个 layer 从 java 传进来,效率太低
JNIEXPORT void JNICALL
Java_com_compilesense_liuyi_caffemodelinput_caffecnn_ConvolutionLayer_testConvolutionLayerCompute(
        JNIEnv *env, jobject instance
){
    LOGD("init test");
    unsigned int stride = 1;
    unsigned int pad = 0;

    ConvolutionLayer::Params params(pad,stride);
    ConvolutionLayer convolutionLayer("testConvolutionLayer",
            params, false);

    float kernel[3*3] = {1,2,3,
                         4,5,6,
                         7,8,9};
    int kernelShape[4] = {1,1,3,3};
    float bias[1] = {0};
    LOGD("setKernel test");
    convolutionLayer.setKernel(kernel, 9,
                            kernelShape, 4,
                            bias, 1);
    float input[5*5] =
            {1,2,3,4,5,
             1,2,3,4,5,
             1,2,3,4,5,
             1,2,3,4,5,
             1,2,3,4,5,};
    std::vector<unsigned int> inputShape(4);
    inputShape[0] = 1;
    inputShape[1] = 1;
    inputShape[2] = 5;
    inputShape[3] = 5;
    float output[9];
    LOGD("compute test");
    convolutionLayer.compute(input, inputShape, output);
    for (int i = 0; i < 9; ++i) {
        LOGD("result:%f",output[i]);
    }

}

#ifdef __cplusplus
}
#endif