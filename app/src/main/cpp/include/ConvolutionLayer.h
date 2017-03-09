//
// Created by 神经元002 on 2017/3/3.
//

#ifndef CAFFEMODELINPUT_CONVOLUTIONLAYER_H
#define CAFFEMODELINPUT_CONVOLUTIONLAYER_H

#include <string>
#include <vector>


class ConvolutionLayer {
private:
    std::string name;

    unsigned int strideSize;
    //步长,步长形状 = {stride_h, stride_w},这个参数可以理解为对 Input 数据进行采样。
    //nnpack 中的 fft算法 不支持 stride 似乎这种情况下 stride 并不能好处。
    unsigned int * stride;
    unsigned int padSize;
    unsigned int * pad; //input 边缘留白,pad的形状 = {pad_h, pad_w}
    int group; //卷积组的大小
    bool nonLinear; //是否有非线性部分

    bool paramHadLoad;
    int weightSize;
    float * weight;
    int * weightShape;
    int weightShapeLength;
    float * bias;
    int biasSize;

public:

    ConvolutionLayer(const std::string &name,
                     unsigned int strideSize, unsigned int *stride,
                     unsigned int padSize, unsigned int *pad,
                     int group, bool nonLinear);
    ~ConvolutionLayer();
    void setParam(float* weight, int weightSize,
                  int* weightShape, int weightShapeLength,
                  float* bias, int biasSize);
    void releaseParams();

    void compute( float * input, std::vector<unsigned  int> inputShape, float * output);
};

#endif //CAFFEMODELINPUT_CONVOLUTIONLAYER_H
