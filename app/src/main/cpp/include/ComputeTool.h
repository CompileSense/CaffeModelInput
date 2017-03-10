//
// Created by 神经元002 on 2017/3/9.
//

#ifndef CAFFEMODELINPUT_COMPUTETOOL_H
#define CAFFEMODELINPUT_COMPUTETOOL_H

#include <vector>
#include <MultiDimData.h>
#include <ConvolutionLayer.h>
#include <utils.h>

bool is_a_ge_zero_and_a_lt_b(int a, int b);

MultiDimData<float > * im2col(
        MultiDimData<float > * inputDataPtr,
        MultiDimData<float > * kernelDataPtr,
        ConvolutionLayer::Params cp);

void relu(MultiDimData<float> *input);

#endif //CAFFEMODELINPUT_COMPUTETOOL_H
