//
// Created by 神经元002 on 2017/3/14.
//

#include <jni.h>
#include <stdlib.h>
#include "MathTool.h"
#include "math.h"
#include "utils.h"

extern "C" {

JNIEXPORT void JNICALL
Java_com_compilesense_liuyi_caffemodelinput_NativeTest_testMathExp(
        JNIEnv *env, jobject instance
){
    float t[1000];
    for (int i = 0; i < 1000; ++i) {
        t[i] = rand()%20;
    }
    float result;

    double st = now_ms();
    for (int i = 0; i < 1000; ++i) {
        result = expf(t[i]);
    }
    LOGD("result1: time%f", now_ms() - st);
    st = now_ms();
    for (int i = 0; i < 1000; ++i) {
        result = expf_C_neon(t[i]);
    }
    LOGD("result2: time%f", now_ms() - st);
}
}