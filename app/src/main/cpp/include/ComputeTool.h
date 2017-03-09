//
// Created by 神经元002 on 2017/3/9.
//

#ifndef CAFFEMODELINPUT_COMPUTETOOL_H
#define CAFFEMODELINPUT_COMPUTETOOL_H

#include <vector>
#include <MultiDimData.h>
#include <ConvolutionLayerParams.h>
#include <utils.h>

bool is_a_ge_zero_and_a_lt_b(int a, int b);
//要注意释放内存
template<typename DataType>
MultiDimData<DataType> * im2col(
        MultiDimData<DataType> * inputDataPtr,
        MultiDimData<DataType> * kernelDataPtr,
        ConvolutionLayerParams cp){
    if (inputDataPtr == NULL || kernelDataPtr == NULL){
        LOGE("错误的输入数据! file:%s, line:%i",__FILE__,__LINE__);
        return NULL;
    }
    MultiDimData<DataType> inputData = *inputDataPtr;
    MultiDimData<DataType> kernelData = *kernelDataPtr;

    if (inputData.num_dimensions != 3 || kernelData.num_dimensions != 2){
        LOGE("错误的输入数据维度! file:%s, line:%i",__FILE__,__LINE__);
    }

    const DataType* data_im = inputData.data_ptr;
    const int channels = (int) inputData.get_c();
    const int h_input = (int) inputData.get_h();
    const int w_input = (int) inputData.get_w();
    const int h_kernel = (int) kernelData.get_h();
    const int w_kernel = (int) kernelData.get_w();
//    LOGE("输入数据维度 channels:%i",channels);
//    LOGE("输入数据维度 h_input:%i",h_input);
//    LOGE("输入数据维度 h_kernel:%i",h_kernel);
    const int pad_h = cp.pad_h;
    const int pad_w = cp.pad_w;
    const int dilation_h = cp.dilation_h;
    const int dilation_w = cp.dilation_w;
    const int stride_h = cp.stride_h;
    const int stride_w = cp.stride_w;
//    LOGE("输入数据维度 pad_h:%i",pad_h);
//    LOGE("输入数据维度 dilation_h:%i",dilation_h);
//    LOGE("输入数据维度 stride_h:%i",stride_h);
//    LOGE("输入数据维度 stride_w:%i",stride_w);


    //先计算输出大小
    const int h_output = (h_input + 2 * pad_h -
                          (dilation_w * (h_kernel - 1) + 1)) / stride_h + 1;
    const int w_output = (w_input + 2 * pad_w -
                          (dilation_w * (w_kernel - 1) + 1)) / stride_w + 1;

    DataType * data_col = new DataType[channels * h_output * w_output];
    std::vector<size_t > data_col_shape(3);
    data_col_shape[0] = (unsigned int) w_output * h_output;
    data_col_shape[1] = (unsigned int) h_kernel * w_kernel;
    data_col_shape[2] = (unsigned int)channels;

    MultiDimData<DataType> * outputDataPtr = new MultiDimData<DataType>(data_col,data_col_shape);

    const int channel_size = h_input * w_input;
    for (int channel = channels; channel--; data_im += channel_size) {
        for (int kernel_row = 0; kernel_row < h_kernel; ++kernel_row){
            for (int kernel_col = 0; kernel_col < w_kernel; ++kernel_col) {
                int input_row =  -pad_h + kernel_row * dilation_h;
                for (int output_rows = h_output; output_rows; output_rows--) {
                    if (!is_a_ge_zero_and_a_lt_b(input_row, h_input)) {
                        for (int output_cols = w_kernel; output_cols; output_cols--) {
                            *(data_col++) = 0;
                        }
                    } else {
                        int input_col = -pad_w + kernel_col * dilation_w;
                        for (int output_col = w_output; output_col; output_col--) {

                            if (is_a_ge_zero_and_a_lt_b(input_col, w_input)) {
                                *(data_col++) = data_im[input_row * w_input + input_col];
                            } else {
                                *(data_col++) = 0;
                            }
                            input_col += stride_w;
                        }
                    }
                    input_row += stride_h;
                }
            }
        }
    }
    return outputDataPtr;
}

#endif //CAFFEMODELINPUT_COMPUTETOOL_H
