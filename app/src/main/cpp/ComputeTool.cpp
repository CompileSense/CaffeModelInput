//
// Created by 神经元002 on 2017/3/9.
//
#include "ComputeTool.h"
#include <arm_neon.h>
#include "math.h"
//const float __tanf_rng[2] = {
//        2.0 / M_PI,
//        M_PI / 2.0
//};
//
//const float __tanf_lut[4] = {
//        -0.00018365f,	//p7
//        -0.16664831f,	//p3
//        +0.00830636f,	//p5
//        +0.99999661f,	//p1
//};
//
//float tanf_c(float x){
//
//    union {
//        float f;
//        int i;
//    } ax, c;
//
//    float r, a, b, xx, cc, cx;
//    int m;
//
//    ax.f = fabsf(x);
//
//    //Range Reduction:
//    m = (int) (ax.f * __tanf_rng[0]);
//    ax.f = ax.f - (((float)m) * __tanf_rng[1]);
//
//    //Test Quadrant
//    ax.f = ax.f - (m & 1) * __tanf_rng[1];
//    ax.i = ax.i ^ ((*(int*)&x) & 0x80000000);
//
//    //Taylor Polynomial (Estrins)
//    xx = ax.f * ax.f;
//    a = (__tanf_lut[0] * ax.f) * xx + (__tanf_lut[2] * ax.f);
//    b = (__tanf_lut[1] * ax.f) * xx + (__tanf_lut[3] * ax.f);
//    xx = xx * xx;
//    r = b + a * xx;
//
//    //cosine
//    c.f = 1.0 - r * r;
//
//    //fast invsqrt approximation (2x newton iterations)
//    cc = c.f;
//    c.i = 0x5F3759DF - (c.i >> 1);		//VRSQRTE
//    cx = cc * c.f;
//    a = (3.0f - cx * c.f) / 2;			//VRSQRTS
//    c.f = c.f * a;
//    cx = cc * c.f;
//    a = (3.0f - cx * c.f) / 2;
//    c.f = c.f * a;
//
//    r = r * c.f;
//
//    return r;
//}


bool is_a_ge_zero_and_a_lt_b(int a, int b) {
    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

MultiDimData<float > * im2col(
        MultiDimData<float > * inputDataPtr,
        MultiDimData<float > * kernelDataPtr,
        ConvolutionLayer::Params cp){
    if (inputDataPtr == NULL || kernelDataPtr == NULL){
        LOGE("错误的输入数据! file:%s, line:%i",__FILE__,__LINE__);
        return NULL;
    }
    MultiDimData<float > inputData = *inputDataPtr;
    MultiDimData<float > kernelData = *kernelDataPtr;

    if (inputData.num_dimensions != 3 || kernelData.num_dimensions != 2){
        LOGE("错误的输入数据维度! file:%s, line:%i",__FILE__,__LINE__);
    }

    const float * data_im = inputData.data_ptr;
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

    float * data_col = new float[channels * h_output * w_output];
    std::vector<size_t > data_col_shape(3);
    data_col_shape[0] = (unsigned int) w_output * h_output;
    data_col_shape[1] = (unsigned int) h_kernel * w_kernel;
    data_col_shape[2] = (unsigned int)channels;

    MultiDimData<float> * outputDataPtr = new MultiDimData<float>(data_col,data_col_shape);

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

void relu(MultiDimData<float> *input){
    if(input == NULL || input->data_ptr == NULL){
        return;
    }
    int num = (int) input->totalSize();
    //一个向量寄存器可128位装4个32位 float。共有16个
    const int numVector = num / 4;
    int left = num % 4;
    int batchIndex = numVector;
    int index = 0;

    float * data = input->data_ptr;
    while(batchIndex > 0){
        batchIndex--;
        float32x4_t vector1 = vld1q_f32(data + index);
        float32x4_t vector1Abs = vabsq_f32(vector1);
        float32x4_t vector1Sum= vaddq_f32(vector1 , vector1Abs);
        float32x4_t result = vmulq_n_f32(vector1Sum, 0.5);
        vst1q_f32(data + index, result);
        index += 4;
    }

    while(left > 0){
        left--;
        float temp = data[index];
        data[index] = (temp > 0) ? temp : 0;
        index++;
    }
}

//ai = 0.25
void prelu(MultiDimData<float> *input){
    const float P = 0.25f;

    if(input == NULL || input->data_ptr == NULL){
        return;
    }
    int num = (int) input->totalSize();
    //一个向量寄存器可128位装4个32位 float。共有16个
    const int numVector = num / 4;
    int left = num % 4;
    int batchIndex = numVector;
    int index = 0;

    float * data = input->data_ptr;
    while(batchIndex > 0){
        batchIndex--;
        float32x4_t vector = vld1q_f32(data + index);//v
        float32x4_t vectorAbs = vabsq_f32(vector);//|v|
        float32x4_t vectorSub = vsubq_f32(vector, vectorAbs);// v - |v|
        float32x4_t vectorTemp1 = vmulq_n_f32(vectorSub, P * 0.5f);//(v - |v|) * 0.5 * ai
        float32x4_t vectorAdd = vaddq_f32(vector, vectorAbs);// v + |v|
        float32x4_t vectorTemp2 = vmulq_n_f32(vectorAdd, 0.5f);
        float32x4_t result = vaddq_f32(vectorTemp1, vectorTemp2);
        vst1q_f32(data + index, result);
        index += 4;
    }

    while(left > 0){
        left--;
        float temp = data[index];
        data[index] = (temp > 0) ? temp : temp * P;
        index++;
    }
}

void tanh(MultiDimData<float> *input){
#ifdef __arm__
    if(input == NULL || input->data_ptr == NULL){
        return;
    }
    int num = (int) input->totalSize();
    float * data = input->data_ptr;
    for (int i = 0; i < num; ++i) {
        data[i] = tanhf(data[i]);
    }
#endif


}

void abs(MultiDimData<float> *input){
    if(input == NULL || input->data_ptr == NULL){
        return;
    }
    int num = (int) input->totalSize();
    //一个向量寄存器可128位装4个32位 float。共有16个
    const int numVector = num / 4;
    int left = num % 4;
    int batchIndex = numVector;
    int index = 0;

    float * data = input->data_ptr;
    while(batchIndex > 0){
        batchIndex--;
        float32x4_t vector1 = vld1q_f32(data + index);
        float32x4_t result = vabsq_f32(vector1);
        vst1q_f32(data + index, result);
        index += 4;
    }

    while(left > 0){
        left--;
        float temp = data[index];
        data[index] = fabsf(temp);
        index++;
    }
}

