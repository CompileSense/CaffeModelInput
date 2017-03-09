//
// Created by 神经元002 on 2017/3/9.
//

#ifndef CAFFEMODELINPUT_CONVOLUTIONLAYERPARAMS_H
#define CAFFEMODELINPUT_CONVOLUTIONLAYERPARAMS_H


class ConvolutionLayerParams {
public:
    int pad,pad_h,pad_w;
    int stride,stride_h,stride_w;
    int dilation,dilation_h,dilation_w; //暂时输入没有


    ConvolutionLayerParams(int pad = 0, int stride = 1, int dilation = 1) {
        pad_h = pad;
        pad_w = pad;
        stride_h = stride;
        stride_w = stride;
        dilation_h = dilation;
        dilation_w = dilation;
    }
};



#endif //CAFFEMODELINPUT_CONVOLUTIONLAYERPARAMS_H
