//
// Created by 神经元002 on 2017/3/3.
//

#ifndef CAFFEMODELINPUT_CONVOLUTIONLAYER_H
#define CAFFEMODELINPUT_CONVOLUTIONLAYER_H

#include <string>
#include <vector>


class ConvolutionLayer {
public:
    class Params {
    public:
        int group;
        int pad,pad_h,pad_w;
        int stride,stride_h,stride_w;
        int dilation,dilation_h,dilation_w; //暂时输入没有

        Params(int pad = 0, int stride = 1, int dilation = 1,int group = 1) {
            pad_h = pad;
            pad_w = pad;
            stride_h = stride;
            stride_w = stride;
            dilation_h = dilation;
            dilation_w = dilation;
        }

        void setParams(int pad_, int stride_, int dilation_, int group_ ){
            pad_h = pad_w = pad = pad_;
            stride_h = stride_w = stride = stride_;
            dilation_h = dilation_w = dilation = dilation_;
            group = group_;
        }
    };

    ConvolutionLayer(const std::string &name, unsigned int stride, unsigned int pad, int group, bool nonLinear);

    ConvolutionLayer(const std::string &name, ConvolutionLayer::Params params,bool nonLinear);

    ~ConvolutionLayer();

    void setKernel(float *weight, int weightSize,
                   int *weightShape, int weightShapeLength,
                   float *bias, int biasSize);

    void releaseKernel();

    void compute( float * input, std::vector<unsigned  int> inputShape, float * output);

private:
    std::string name;
    bool nonLinear; //是否有非线性部分
    ConvolutionLayer::Params params;
    bool paramHadLoad;
    int weightSize;
    float * weight;
    int * weightShape;
    int weightShapeLength;
    float * bias;
    int biasSize;
};

#endif //CAFFEMODELINPUT_CONVOLUTIONLAYER_H
