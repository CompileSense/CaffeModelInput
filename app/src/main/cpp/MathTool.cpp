//
// Created by 神经元002 on 2017/3/14.
//



#include "MathTool.h"
#include "arm_neon.h"
#include "utils.h"
#ifdef __cplusplus
extern "C" {
#endif

const float __expf_rng[2] = {
        1.442695041f,
        0.693147180f
};

const float __expf_lut[8] = {
        0.9999999916728642f,        //p0
        0.04165989275009526f,    //p4
        0.5000006143673624f,    //p2
        0.0014122663401803872f,    //p6
        1.000000059694879f,        //p1
        0.008336936973260111f,    //p5
        0.16666570253074878f,    //p3
        0.00019578093328483123f    //p7
};

float expf_neon__hf(float x) {
#ifdef __MATH_NEON
    asm volatile (
    "vdup.f32 		d0, d0[0]				\n\t"    //d0 = {x, x}

    //Range Reduction:
    "vld1.32 		d2, [%0]				\n\t"    //d2 = {invrange, range}
    "vmul.f32 		d6, d0, d2[0]			\n\t"    //d6 = d0 * d2[0]
    "vcvt.s32.f32 	d6, d6					\n\t"    //d6 = (int) d6
    "vcvt.f32.s32 	d1, d6					\n\t"    //d1 = (float) d6
    "vmls.f32 		d0, d1, d2[1]			\n\t"    //d0 = d0 - d1 * d2[1]

    //polynomial:
    "vmul.f32 		d1, d0, d0				\n\t"    //d1 = d0*d0 = {x^2, x^2}
    "vld1.32 		{d2, d3, d4, d5}, [%1]	\n\t"    //q1 = {p0, p4, p2, p6}, q2 = {p1, p5, p3, p7} ;
    "vmla.f32 		q1, q2, d0[0]			\n\t"    //q1 = q1 + q2 * d0[0]
    "vmla.f32 		d2, d3, d1[0]			\n\t"    //d2 = d2 + d3 * d1[0]
    "vmul.f32 		d1, d1, d1				\n\t"    //d1 = d1 * d1 = {x^4, x^4}
    "vmla.f32 		d2, d1, d2[1]			\n\t"    //d2 = d2 + d1 * d2[1]

            //multiply by 2 ^ m
    "vshl.i32 		d6, d6, #23				\n\t"    //d6 = d6 << 23
    "vadd.i32 		d0, d2, d6				\n\t"    //d0 = d2 + d6

    ::"r"(__expf_rng), "r"(__expf_lut)
: "d0", "d1", "q1", "q2", "d6"
    );
#endif
}

float expf_neon__(float x) {
#ifdef __MATH_NEON
    asm volatile ("vmov.f32 s0, r0 		\n\t");
    expf_neon__hf(x);
    asm volatile ("vmov.f32 r0, s0 		\n\t");
#else
    return expf_c__(x);
#endif
} ;

float expf_c__(float x) {
    float a, b, c, d, xx;
    int m;

    union {
        float f;
        int i;
    } r;

    //Range Reduction:
    m = (int) (x * __expf_rng[0]);
    x = x - ((float) m) * __expf_rng[1];
    //Taylor Polynomial (Estrins)
    a = (__expf_lut[4] * x) + (__expf_lut[0]);
    c = (__expf_lut[5] * x) + (__expf_lut[1]);
    b = (__expf_lut[6] * x) + (__expf_lut[2]);
    d = (__expf_lut[7] * x) + (__expf_lut[3]);
//
//    LOGD("a:%f",a);
//    LOGD("b:%f",b);
//    LOGD("c:%f",c);
//    LOGD("d:%f",d);


    xx = x * x;

    a = a + b * xx;
    c = c + d * xx;

//    LOGD("a2:%f",a);
//    LOGD("c2:%f",c);
    xx = xx * xx;
//    LOGD("xx2:%f",xx);
    r.f = a + c * xx;
    //multiply by 2 ^ m
    m = m << 23;
    r.i = r.i + m;
//    LOGD("r.f:%f",r.f);
//    LOGD("r.i:%i",r.i);
    return r.f;
}
const float lut1[4] = {__expf_lut[0],__expf_lut[1],__expf_lut[2],__expf_lut[3]};//{a,c,b,d}
const float lut2[4] = {__expf_lut[4],__expf_lut[5],__expf_lut[6],__expf_lut[7]};//{a,c,b,d}

float expf_C_neon(float x){

    union {
        float f;
        int i;
    } r;

    //Range Reduction:
    int m = (int) (x * __expf_rng[0]);
    x = x - ((float) m) * __expf_rng[1];
    float xx = x*x;
    float xx2 = xx*xx;

    float32x4_t vlut1 = vld1q_f32(lut1);
    float32x4_t vlut2 = vld1q_f32(lut2);

    float32x4_t vx = vdupq_n_f32(x);
    float32x2_t vxx = vdup_n_f32(xx);

    float32x4_t v_acbd = vmlaq_f32(vlut1, vx, vlut2);//{a,c,b,d}
//    float32_t a = vgetq_lane_f32(v_acbd,0);
//    float32_t c = vgetq_lane_f32(v_acbd,1);
//    float32_t b = vgetq_lane_f32(v_acbd,2);
//    float32_t d = vgetq_lane_f32(v_acbd,3);
//    LOGD("a:%f",a);
//    LOGD("b:%f",b);
//    LOGD("c:%f",c);
//    LOGD("d:%f",d);


    float32x2_t v_bd = vget_high_f32(v_acbd);
    float32x2_t v_ac = vget_low_f32(v_acbd);
    float32x2_t v_acMla = vmla_f32(v_ac, v_bd, vxx);
    float32_t a = vget_lane_f32(v_acMla,0);
    float32_t c = vget_lane_f32(v_acMla,1);
    r.f = a + c * xx2;


    m = m << 23;
    r.i = r.i + m;
    return r.f;
}

#ifdef __cplusplus
}
#endif