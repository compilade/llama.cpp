#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu/ggml-cpu-impl.h"
#include "ggml-cpu.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h> // for qsort
#include <stdio.h>  // for GGML_ASSERT

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define GROUP_MAX_EPS_IQ2_S 1e-8f
#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define GROUP_MAX_EPS_IQ1_S 1e-12f

#define UNUSED GGML_UNUSED

static inline int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

// reference implementation for deterministic creation of model files
void quantize_row_q4_0_ref(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -8;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

void quantize_row_q4_1_ref(const float * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k) {
    const int qk = QK4_1;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float min = FLT_MAX;
        float max = -FLT_MAX;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];

            if (v < min) min = v;
            if (v > max) max = v;
        }

        const float d  = (max - min) / ((1 << 4) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);
        y[i].m = GGML_FP32_TO_FP16(min);

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = (x[i*qk + 0    + j] - min)*id;
            const float x1 = (x[i*qk + qk/2 + j] - min)*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

void quantize_row_q5_0_ref(const float * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK5_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -16;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        uint32_t qh = 0;

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
            const uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

            y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            // get the 5-th bit and store it in qh at the right position
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + qk/2);
        }

        memcpy(&y[i].qh, &qh, sizeof(qh));
    }
}

void quantize_row_q5_1_ref(const float * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k) {
    const int qk = QK5_1;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float min = FLT_MAX;
        float max = -FLT_MAX;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];

            if (v < min) min = v;
            if (v > max) max = v;
        }

        const float d  = (max - min) / ((1 << 5) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);
        y[i].m = GGML_FP32_TO_FP16(min);

        uint32_t qh = 0;

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = (x[i*qk + 0    + j] - min)*id;
            const float x1 = (x[i*qk + qk/2 + j] - min)*id;

            const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
            const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

            y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            // get the 5-th bit and store it in qh at the right position
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + qk/2);
        }

        memcpy(&y[i].qh, &qh, sizeof(y[i].qh));
    }
}

// reference implementation for deterministic creation of model files
void quantize_row_q8_0_ref(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_0; j++) {
            const float v = x[i*QK8_0 + j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = x[i*QK8_0 + j]*id;

            y[i].qs[j] = roundf(x0);
        }
    }
}

// reference implementation for deterministic creation of model files
void quantize_row_q8_1_ref(const float * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k) {
    assert(QK8_1 == 32);
    assert(k % QK8_1 == 0);
    const int nb = k / QK8_1;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_1; j++) {
            const float v = x[i*QK8_1 + j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        int sum = 0;

        for (int j = 0; j < QK8_1/2; ++j) {
            const float v0 = x[i*QK8_1           + j]*id;
            const float v1 = x[i*QK8_1 + QK8_1/2 + j]*id;

            y[i].qs[          j] = roundf(v0);
            y[i].qs[QK8_1/2 + j] = roundf(v1);

            sum += y[i].qs[          j];
            sum += y[i].qs[QK8_1/2 + j];
        }

        y[i].s = GGML_FP32_TO_FP16(sum*d);
    }
}

static inline int best_index_mxfp4(float x, float e) {
    int best_index = 0;
    float best_err = fabsf(kvalues_mxfp4[0]*e - x);
    for (int i = 1; i < 16; i++) {
        float err = fabsf(kvalues_mxfp4[i]*e - x);
        if (err < best_err) {
            best_index = i;
            best_err = err;
        }
    }
    return best_index;
}

void quantize_row_mxfp4_ref(const float * GGML_RESTRICT x, block_mxfp4 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK_MXFP4;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];

            if (amax < fabsf(v)) {
                amax = fabsf(v);
            }
        }

        const uint8_t e = amax > 0.0f ? (uint8_t) (floorf(log2f(amax)) - 2 + 127) : 0;

        const float d = GGML_E8M0_TO_FP32_HALF(e);

        y[i].e = e;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t x0 = best_index_mxfp4(x[i*qk + 0    + j], d);
            const uint8_t x1 = best_index_mxfp4(x[i*qk + qk/2 + j], d);

            y[i].qs[j]  = x0;
            y[i].qs[j] |= x1 << 4;
        }
    }
}

void dequantize_row_q4_0(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

void dequantize_row_q4_1(const block_q4_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK4_1;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float m = GGML_FP16_TO_FP32(x[i].m);

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F);
            const int x1 = (x[i].qs[j] >>   4);

            y[i*qk + j + 0   ] = x0*d + m;
            y[i*qk + j + qk/2] = x1*d + m;
        }
    }
}

void dequantize_row_q5_0(const block_q5_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK5_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((x[i].qs[j] >>   4) | xh_1) - 16;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

void dequantize_row_q5_1(const block_q5_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK5_1;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float m = GGML_FP16_TO_FP32(x[i].m);

        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            const int x0 = (x[i].qs[j] & 0x0F) | xh_0;
            const int x1 = (x[i].qs[j] >>   4) | xh_1;

            y[i*qk + j + 0   ] = x0*d + m;
            y[i*qk + j + qk/2] = x1*d + m;
        }
    }
}

void dequantize_row_q8_0(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK8_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }
}

void dequantize_row_mxfp4(const block_mxfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK_MXFP4;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_E8M0_TO_FP32_HALF(x[i].e);

        for (int j = 0; j < qk/2; ++j) {
            const int8_t x0 = kvalues_mxfp4[x[i].qs[j] & 0x0F];
            const int8_t x1 = kvalues_mxfp4[x[i].qs[j] >>   4];

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

//
// 2-6 bit quantization in super-blocks
//

//
// ===================== Helper functions
//
static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static float make_qx_quants(int n, int nmax, const float * GGML_RESTRICT x, int8_t * GGML_RESTRICT L, int rmse_type,
        const float * GGML_RESTRICT qw) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < GROUP_MAX_EPS) { // all zero
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
        }
        return 1/iscale;
    }
    bool return_early = false;
    if (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }
    float sumlx = 0;
    float suml2 = 0;
#ifdef HAVE_BUGGY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
    for (volatile int i = 0; i < n; ++i) {
#else
    for (int i = 0; i < n; ++i) {
#endif
        int l = nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
        float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    float scale = suml2 ? sumlx/suml2 : 0.0f;
    if (return_early) return suml2 > 0 ? 0.5f*(scale + 1/iscale) : 1/iscale;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) {
            continue;
        }
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
        }
    }
    return scale;
}

static float make_q3_quants(int n, int nmax, const float * GGML_RESTRICT x, int8_t * GGML_RESTRICT L, bool do_rmse) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < GROUP_MAX_EPS) { // all zero
        for (int i = 0; i < n; ++i) { L[i] = 0; }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (do_rmse) {
        float sumlx = 0;
        float suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            L[i] = l;
            float w = x[i]*x[i];
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        for (int itry = 0; itry < 5; ++itry) {
            int n_changed = 0;
            for (int i = 0; i < n; ++i) {
                float w = x[i]*x[i];
                float slx = sumlx - w*x[i]*L[i];
                if (slx > 0) {
                    float sl2 = suml2 - w*L[i]*L[i];
                    int new_l = nearest_int(x[i] * sl2 / slx);
                    new_l = MAX(-nmax, MIN(nmax-1, new_l));
                    if (new_l != L[i]) {
                        slx += w*x[i]*new_l;
                        sl2 += w*new_l*new_l;
                        if (sl2 > 0 && slx*slx*suml2 > sumlx*sumlx*sl2) {
                            L[i] = new_l; sumlx = slx; suml2 = sl2;
                            ++n_changed;
                        }
                    }
                }
            }
            if (!n_changed) {
                break;
            }
        }
        for (int i = 0; i < n; ++i) {
            L[i] += nmax;
        }
        return sumlx / suml2;
    }
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
    }
    return 1/iscale;
}

static float make_qkx1_quants(int n, int nmax, const float * GGML_RESTRICT x, uint8_t * GGML_RESTRICT L, float * GGML_RESTRICT the_min,
        int ntry, float alpha) {
    float min = x[0];
    float max = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
    }
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = 0;
        return 0.f;
    }
    if (min > 0) min = 0;
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    for (int itry = 0; itry < ntry; ++itry) {
        float sumlx = 0; int suml2 = 0;
        bool did_change = false;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = MAX(0, MIN(nmax, l));
            if (l != L[i]) {
                L[i] = l;
                did_change = true;
            }
            sumlx += (x[i] - min)*l;
            suml2 += l*l;
        }
        scale = sumlx/suml2;
        float sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += x[i] - scale*L[i];
        }
        min = alpha*min + (1 - alpha)*sum/n;
        if (min > 0) min = 0;
        iscale = 1/scale;
        if (!did_change) break;
    }
    *the_min = -min;
    return scale;
}

static float make_qkx2_quants(int n, int nmax, const float * GGML_RESTRICT x, const float * GGML_RESTRICT weights,
        uint8_t * GGML_RESTRICT L, float * GGML_RESTRICT the_min, uint8_t * GGML_RESTRICT Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
#ifdef HAVE_BUGGY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
    for (volatile int i = 1; i < n; ++i) {
#else
    for (int i = 1; i < n; ++i) {
#endif
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    float best_error = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale*(x[i] - min));
        L[i] = MAX(0, MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        best_error += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = MAX(0, MIN(nmax, l));
            Laux[i] = l;
            float w = weights[i];
            sum_l += w*l;
            sum_l2 += w*l*l;
            sum_xl += w*l*x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                float w = weights[i];
                cur_error += w * diff;
            }
            if (cur_error < best_error) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

struct fraction {
    float numer;
    float denom;
    int i;
};

// comparator function for sorting fractions
static inline int compare_fractions_desc(const void * a, const void * b) {
    const struct fraction * f_a = (const struct fraction *) a;
    const struct fraction * f_b = (const struct fraction *) b;
    float na = f_a->numer;
    float da = f_a->denom;
    float nb = f_b->numer;
    float db = f_b->denom;

    // Stable sort
    // a - b sorts ascending, which means
    // 1 swaps, -1 stays
    if (da == db) { // equal denominators
        return (na == nb) ? ((a > b) ? 1 : -1) : (na < nb) ? 1 : -1;
    }
    if (na == nb) { // equal numerators
        return (da > db) ? 1 : -1;
    }
    float ab = na * db;
    float ba = nb * da;
    return (ab == ba) ? ((a > b) ? 1 : -1) : (ab < ba) ? 1 : -1;
}

struct k_heap_cell {
    float frac;
    float x;
    int x_i;
    int k_i;
};

// Faster enumeration for cumulative search
struct k_heap {
    int n;
    int k;
    int mid_k;
    int8_t kmin;
    int8_t kmax;
    int8_t kmax_step;
    const float * odd;
    const float * steps;
    // TODO: fixed-size inline array for the indices
    const uint8_t * indices; // for faster non-linear rounding
    const int8_t * kvalues;
    struct k_heap_cell * heap;
};

// build a max heap out of k_cells starting from node i;
// makes sure the node i is bigger than its children
static void k_heap_build(struct k_heap * heap, int i) {
    const int n = heap->n;
    int max = i;
    int prev_max;
    while (max < n / 2) {
        prev_max = max;
        const int left = 2*(max + 1) - 1;
        const int right = 2*(max + 1);
        if (left < n && heap->heap[left].frac > heap->heap[max].frac) {
            max = left;
        }
        if (right < n && heap->heap[right].frac > heap->heap[max].frac) {
            max = right;
        }
        if (max != prev_max) {
            const struct k_heap_cell tmp = heap->heap[prev_max];
            heap->heap[prev_max] = heap->heap[max];
            heap->heap[max] = tmp;
        } else {
            break;
        }
    }
}

// Assuming kvalues is sorted from most negative to most positive.
// odd and steps are buffers of size k, while heap_cells should be big enough for x later on.
static void k_heap_init(struct k_heap * GGML_RESTRICT k_heap, int k, const int8_t * GGML_RESTRICT kvalues, struct k_heap_cell * GGML_RESTRICT heap_cells, float * GGML_RESTRICT odd, float * GGML_RESTRICT steps, uint8_t * GGML_RESTRICT indices) {
    GGML_ASSERT(k_heap && kvalues && heap_cells && odd);
    k_heap->n = 0;
    k_heap->k = k;
    k_heap->odd = odd;
    k_heap->steps = steps;
    k_heap->indices = indices;
    k_heap->kvalues = kvalues;
    k_heap->heap = heap_cells;

    int amin = abs(kvalues[0]);
    int amax = abs(kvalues[0]);
    int mid_k = 0;
    int max_k = 0;
    for (int i = 1; i < k; ++i) {
        const int ak = abs(kvalues[i]);
        if (ak < amin) {
            amin = ak;
            mid_k = i;
        }
        if (ak > amax) {
            amax = ak;
            max_k = i;
        }
    }
    k_heap->mid_k = mid_k;
    k_heap->kmin = kvalues[mid_k];
    k_heap->kmax = kvalues[max_k];
    k_heap->kmax_step = abs(max_k < mid_k ? kvalues[max_k + 1] - k_heap->kmax : k_heap->kmax - kvalues[max_k - 1]);

    for (int i = 0; i < k - 1; ++i) {
        const float threshold = kvalues[i + 1] + kvalues[i];
        const float step      = kvalues[i + 1] - kvalues[i];
        // It's amazing how their product is the difference between consecutive squares of the kvalues,
        // but it makes sense because a*a - b*b == (a + b)*(a - b).
        GGML_ASSERT(threshold * step != 0.0f);
        odd[i + (i >= mid_k ? 1 : 0)]   = fabsf(threshold);
        steps[i + (i >= mid_k ? 1 : 0)] = fabsf(step);
    }
    odd[mid_k] = 0.0f;
    steps[mid_k] = 0.0f;

    {
        int cur_k = 0;
        int cur   = (int) kvalues[cur_k] * 2;
        int next  = (int) kvalues[cur_k + 1] * 2;  // NOTE: this assumes k is at least 2
        for (int i = -256; i < 254; ++i) {
            // TODO: is this always correct?
            if (next != cur && abs(i - next) <= abs(i - cur)) {
                cur = next;
                cur_k += 1;
                if (cur_k < k) {
                    next = (int) kvalues[cur_k + 1] * 2;
                }
            }
            indices[i + 256] = cur_k;
        }
    }
}

// for linear types which have a constant step of 1 between representable values
static void k_heap_init_linear(struct k_heap * k_heap, int nmin, int nmax, struct k_heap_cell * GGML_RESTRICT heap_cells, float * GGML_RESTRICT odd) {
    GGML_ASSERT(k_heap && heap_cells && odd);
    nmin = MIN(0, nmin);
    nmax = MAX(0, nmax);
    k_heap->n = 0;
    k_heap->k = nmax - nmin + 1;
    k_heap->odd = odd;
    k_heap->steps = NULL;
    k_heap->indices = NULL;
    k_heap->kvalues = NULL;
    k_heap->heap = heap_cells;

    k_heap->mid_k = -nmin;
    k_heap->kmin = 0; // the range should always overlap 0
    k_heap->kmax = abs(nmin) > abs(nmax) ? nmin : nmax;
    k_heap->kmax_step = 1;

    for (int i = nmin; i < nmax; ++i) {
        // odd numbers are the difference between consecutive squares
        odd[i - nmin + (i >= 0 ? 1 : 0)] = fabsf((float) (i + (i + 1)));
    }
    odd[-nmin] = 0.0f;
}

// with initial quantized values
static void k_heap_set_x_L(struct k_heap * k_heap, const float * GGML_RESTRICT x, const int8_t * GGML_RESTRICT L, int n, bool invert_sign) {
    int j = 0;
    for (int i = 0; i < n; ++i) {
        const int k_i = ((x[i] < 0.0f) != invert_sign) ? L[i] - 1 : L[i] + 1;
        GGML_ASSERT(k_i != k_heap->mid_k);
        if (k_i >= 0 && k_i < k_heap->k) {
            k_heap->heap[j++] = (struct k_heap_cell) {
                .k_i=k_i,
                .x_i=i,
                .x=fabsf(x[i]),
                .frac=fabsf(x[i] / k_heap->odd[k_i]),
            };
        }
    }
    k_heap->n = j;

    for (int i = (k_heap->n / 2) - 1; i >= 0; --i) {
        k_heap_build(k_heap, i);
    }
}

// assuming the initial quantized value are all at k_heap->mid_k
static void k_heap_set_x(struct k_heap * k_heap, const float * GGML_RESTRICT x, int n, bool invert_sign) {
    int j = 0;
    for (int i = 0; i < n; ++i) {
        const int k_i = ((x[i] < 0.0f) != invert_sign) ? k_heap->mid_k - 1 : k_heap->mid_k + 1;
        if (k_i >= 0 && k_i < k_heap->k) {
            k_heap->heap[j++] = (struct k_heap_cell) {
                .k_i=k_i,
                .x_i=i,
                .x=fabsf(x[i]),
                .frac=fabsf(x[i] / k_heap->odd[k_i]),
            };
        }
    }
    k_heap->n = j;

    for (int i = (k_heap->n / 2) - 1; i >= 0; --i) {
        k_heap_build(k_heap, i);
    }
}

// returns the closest quantized value
static inline int k_heap_best_index(const struct k_heap * k_heap, float x) {
    GGML_ASSERT(k_heap->indices);
    if (x <= -128.0f) {
        return 0;
    }
    if (x >= 127.0f) {
        return k_heap->k - 1;
    }
    // (-256 to 253) --> (0 to 509)
    // int i = (int)floorf(x) + lroundf(x) + 256;
    // NOTE: using faster primitives for rounding
    int i = (int)(x + 128.0f) + nearest_int(x) + 128;
    return k_heap->indices[i];
}

// returns the fractions in descending order
static struct fraction k_heap_pop(struct k_heap * k_heap) {
    if (k_heap && k_heap->n > 0) {
        struct k_heap_cell * heap_cell = k_heap->heap;
        struct fraction frac;
        if (k_heap->steps) {
            const float step = k_heap->steps[heap_cell->k_i];
            // Properly turn this into a difference of consecutive squares even for non-linear steps
            frac = (struct fraction) {
                .numer=heap_cell->x * step,
                .denom=k_heap->odd[heap_cell->k_i] * step,
                .i=heap_cell->x_i,
            };
        } else {
            // step is always 1 for linear quants
            frac = (struct fraction) {
                .numer=heap_cell->x,
                .denom=k_heap->odd[heap_cell->k_i],
                .i=heap_cell->x_i,
            };
        }

        if (heap_cell->k_i < k_heap->mid_k) {
            if (heap_cell->k_i > 0) {
                heap_cell->k_i -= 1;
                heap_cell->frac = heap_cell->x / k_heap->odd[heap_cell->k_i];
            } else {
                // remove this node
                k_heap->heap[0] = k_heap->heap[k_heap->n - 1];
                k_heap->n -= 1;
            }
        } else {
            if (heap_cell->k_i + 1 < k_heap->k) {
                heap_cell->k_i += 1;
                heap_cell->frac = heap_cell->x / k_heap->odd[heap_cell->k_i];
            } else {
                // remove this node
                k_heap->heap[0] = k_heap->heap[k_heap->n - 1];
                k_heap->n -= 1;
            }
        }
        k_heap_build(k_heap, 0);

        return frac;
    }
    return (struct fraction) {
        .numer=0.0f,
        .denom=0.0f,
        .i=-1,
    };
}

// Fast sorting of scales with a hybrid non-comparative sort.
//
// This allows incrementally calculating error (nearly) exhaustively
// for all scales which result in distinct linear or non-linear quantizations.
struct k_sort {
    int n;
    int k;  // number of k_values

    // some useful info about the k_values
    int8_t kmin;   // absmin k_value (but with its sign)
    int8_t kmax;   // absmax k_value (but with its sign)
    int8_t mid_k;  // id of kmin into kvalues

    // These have size k
    const int8_t * k_values;  // if NULL, it's assumed to be linear (i - mid_k)
    float *  odd;       // k_values[i + 1] + k_values[i] (odd numbers when linear, hence the name)
    float *  step;      // k_values[i + 1] - k_values[i] (if NULL, assumed to be 1)

    // All of the below arrays need to have size n at least.
    int32_t * ids;      // original ids (into the full-precision block)
    int32_t * k_ids;    // denominator ids (into odd and step)
    int32_t * aux_ids;  // argsort ids;
    float *   frac;     // what is actually being sorted

    // temporary buffer when sorting the other buffers
    union {
        float *   aux_f;
        int32_t * aux_i;
    };

    // Holds indices into the bucket counts
    uint16_t * Iaux;
    // Where the histogram will be counted
    // TODO: experiment with different bucket sizes than n
    uint16_t * buckets;

    // For faster non-linear rounding, always 510 bytes in size
    // TODO: static buffer, but how to not include it for non-linear quants?
    int8_t * k_indices;
};

// helper for k_sort buffer sizes
#define K_SORT_BUF_SIZE(n, k, range, nl) ( \
    (/* odd, step */ (k) * (sizeof(float) * (1 + !!(nl)))) + \
    (/* ids, k_ids, aux_ids, frac, aux */ ((n) * (range)) * (sizeof(int32_t) * 3 + sizeof(float) * 2)) + \
    (/* Iaux, buckets */ ((n) * (range) * (sizeof(uint16_t) * 2))) + \
    (/* k_indices */ ((nl) ? 510 * sizeof(int8_t) : 0)) \
)

// For non-linear quants.
// k is the number of possible k-values,
// range is the longest number of k-values starting from the middle one,
// block is the size of a block.
#define K_SORT_BUF_SIZE_NL(block, k, range) (K_SORT_BUF_SIZE((block), (k), (range), 1))

// For linear quants. nmin should be <= 0, and nmax >= 0. block is the size of a block.
#define K_SORT_BUF_SIZE_LINEAR(block, nmin, nmax) (K_SORT_BUF_SIZE((block), (nmax) - (nmin) + 1, (nmax) > -(nmin) ? (nmax) : -(nmin), 0))

// for non-linear quants
// TODO: maybe use an array of structs instead, or malloc to simplify initialization
static void k_sort_init(struct k_sort * s, int n, int k, const int8_t * kvalues, uint8_t * buf) {
    s->n = 0;
    s->k = k;

    s->k_values = kvalues;
    s->odd      = (float *) (buf);
    s->step     = (float *) (buf + k * sizeof(float));

    buf += (2 * k) * sizeof(float);

    int k_amin = abs(kvalues[0]);
    int k_amax = abs(kvalues[0]);
    int mid_k  = 0;
    int max_k  = 0;
    for (int i = 1; i < k; ++i) {
        const int ak = abs(kvalues[i]);
        if (ak < k_amin) { k_amin = ak; mid_k = i; }
        if (ak > k_amax) { k_amax = ak; max_k = i; }
    }

    const int max_range = (mid_k > (k - mid_k - 1)) ? mid_k : k - mid_k - 1;

    s->ids     = (int32_t *) (buf + max_range * n * (sizeof(int32_t) * 0));
    s->k_ids   = (int32_t *) (buf + max_range * n * (sizeof(int32_t) * 1));
    s->aux_ids = (int32_t *) (buf + max_range * n * (sizeof(int32_t) * 2));
    s->frac    = (float *) (buf + max_range * n * (sizeof(int32_t) * 3));
    s->aux_f   = (float *) (buf + max_range * n * (sizeof(int32_t) * 3 + sizeof(float)));

    buf += max_range * n * (sizeof(int32_t) * 3 + sizeof(float) * 2);

    s->Iaux    = (uint16_t *) (buf);
    s->buckets = (uint16_t *) (buf + n * max_range * sizeof(uint16_t));

    buf += 2 * n * max_range * sizeof(uint16_t);

    s->k_indices = (int8_t *) buf;

    for (int i = 1; i < k; ++i) {
        // 0 to k - 1, skipping mid_k
        const int j = i - ((int) (i <= mid_k));

        s->odd[j]  = abs(kvalues[i] + kvalues[i - 1]);
        s->step[j] = abs(kvalues[i] - kvalues[i - 1]);
    }
    s->odd[mid_k]  = 1.0f;
    s->step[mid_k] = 1.0f;

    s->kmin  = kvalues[mid_k];
    s->kmax  = kvalues[max_k];
    s->mid_k = mid_k;

    // for faster non-linear rounding
    {
        int cur_k = 0;
        int cur   = (int) kvalues[cur_k] * 2;
        int next  = (int) kvalues[cur_k + 1] * 2;  // assuming k is at least 2
        for (int i = -256; i < 254; ++i) {
            // TODO: is this always correct?
            if (next != cur && abs(i - next) <= abs(i - cur)) {
                cur = next;
                cur_k += 1;
                if (cur_k + 1 < k) {
                    next = (int) kvalues[cur_k + 1] * 2;
                }
            }
            s->k_indices[i + 256] = cur_k;
        }
    }
}

// buf should have size from K_SORT_BUF_SIZE_LINEAR(n, nmin, nmax)
static void k_sort_init_linear(struct k_sort * s, int n, int nmin, int nmax, uint8_t * buf) {
    nmin = MIN(0, nmin);
    nmax = MAX(0, nmax);

    const int max_range = (nmax > -nmin ? nmax : -nmin);

    s->n     = 0;
    s->k     = nmax - nmin + 1;
    s->mid_k = -nmin;
    s->kmin  = 0;
    s->kmax  = -nmin > nmax ? nmin : nmax;

    s->k_values = NULL;
    s->odd      = (float *) (buf);
    s->step     = NULL;

    buf += s->k * sizeof(float);

    s->ids     = (int32_t *) (buf + max_range * n * (sizeof(int32_t) * 0));
    s->k_ids   = (int32_t *) (buf + max_range * n * (sizeof(int32_t) * 1));
    s->aux_ids = (int32_t *) (buf + max_range * n * (sizeof(int32_t) * 2));
    s->frac    = (float *) (buf + max_range * n * (sizeof(int32_t) * 3));
    s->aux_f   = (float *) (buf + max_range * n * (sizeof(int32_t) * 3 + sizeof(float)));

    buf += max_range * n * (sizeof(int32_t) * 3 + sizeof(float) * 2);

    s->Iaux    = (uint16_t *) (buf);
    s->buckets = (uint16_t *) (buf + n * max_range * sizeof(uint16_t));

    s->k_indices = NULL;

    for (int i = nmin; i < nmax; ++i) {
        const int j = i - nmin + (i >= 0);

        s->odd[j] = abs(i + (i + 1));
    }
    s->odd[-nmin] = 1.0f;
}

static inline int k_sort_best_index(struct k_sort * s, float x) {
    if (x <= -128.0f) {
        return 0;
    }
    if (x >= 127.0f) {
        return s->k - 1;
    }
    // (-256 to 253) --> (0 to 509)
    // const int i = (int)floorf(x) + lroundf(x) + 256;
    // NOTE: using faster primitives for rounding
    const int i = (int) (x + 128.0f) + nearest_int(x) + 128;
    return s->k_indices[i];
}

// Interpolation sort using an hybrid of non-comparative counting sort and insertion sort.
static void k_sort_frac_descending(struct k_sort * s) {
    const int N_BUCKETS = s->n;
    memset(s->buckets, 0, N_BUCKETS * sizeof(*(s->buckets)));

    float max_frac = s->frac[0];
    float min_frac = max_frac;
    for (int i = 1; i < s->n; ++i) {
        const float f = s->frac[i];
        if (f > max_frac) { max_frac = f; }
        if (f < min_frac) { min_frac = f; }
    }

    if (max_frac - min_frac > GROUP_MAX_EPS) {
        const float iscale = (N_BUCKETS - 1) / (max_frac - min_frac);
        // Counting sort (descending)
        // This partially sorts the values and works best for uniform distributions.
        for (int i = 0; i < s->n; ++i) {
            const int j = N_BUCKETS - 1 - MAX(0, MIN(nearest_int((s->frac[i] - min_frac) * iscale), N_BUCKETS - 1));
            s->buckets[j] += 1;
            s->Iaux[i] = j;
        }

        for (int j = 1; j < N_BUCKETS; ++j) {
            s->buckets[j] += s->buckets[j - 1];
        }

        for (int i = s->n - 1; i >= 0; --i) {
            const int l = s->Iaux[i];
            const int j = --(s->buckets[l]);
            s->aux_ids[j] = i;
            s->aux_f[j] = s->frac[i];
        }

        { float * tmp = s->frac; s->frac = s->aux_f; s->aux_f = tmp; }

        for (int i = 0; i < s->n; ++i) {
            const int j = s->aux_ids[i];
            s->aux_i[i] = s->k_ids[j];
        }

        { int32_t * tmp = s->k_ids; s->k_ids = s->aux_i; s->aux_i = tmp; }

        for (int i = 0; i < s->n; ++i) {
            const int j = s->aux_ids[i];
            s->aux_i[i] = s->ids[j];
        }

        { int32_t * tmp = s->ids; s->ids = s->aux_i; s->aux_i = tmp; }
    }

    // Insertion sort (descending)
    // This is very fast on mostly-sorted data,
    // but will be slow if everything ended up
    // in a single bucket in the previous step.
    // TODO: use another adaptive sort algorithm with a better worst case time complexity
    for (int i = 1; i < s->n; ++i) {
        const float   tmp      = s->frac[i];
        const int32_t tmp_k_id = s->k_ids[i];
        const int32_t tmp_id   = s->ids[i];

        int j = i;
        for (; j > 0 && s->frac[j - 1] < tmp; --j) {
            s->frac[j]  = s->frac[j - 1];
            s->k_ids[j] = s->k_ids[j - 1];
            s->ids[j]   = s->ids[j - 1];
        }
        if (j != i) {
            s->frac[j]  = tmp;
            s->k_ids[j] = tmp_k_id;
            s->ids[j]   = tmp_id;
        }
    }
}

static void k_sort_set_x_L(struct k_sort * s, int n, int w_amax_i, const float * GGML_RESTRICT x,
                           const int8_t * GGML_RESTRICT L, bool negative_scale) {
    const float wmax = fabsf(x[w_amax_i]);
    const int k = s->k;
    // Extrapolate the extremities (assuming k is at least 2)
    const float max_odd = (x[w_amax_i] < 0.0f) != negative_scale ? s->odd[0] + fabsf(s->odd[0] - s->odd[1]) :
                                                                 s->odd[k - 1] + fabsf(s->odd[k - 1] - s->odd[k - 2]);
    int m = 0;
    for (int i = 0; i < n; ++i) {
        if (x[i] == 0.0f) { continue; }

        const float v = fabsf(x[i]);
        const float v_max_odd = v * max_odd;

        const int odd_dir = (x[i] < 0.0f) != negative_scale ? -1 : 1;

        for (int j = L[i] + odd_dir; 0 <= j && j < s->k; j += odd_dir) {
            const float odd = s->odd[j];

            // Only include scales which would not clamp the "most important" value
            if (wmax * odd < v_max_odd) {
                s->frac[m]  = v / odd;
                s->ids[m]   = i;
                s->k_ids[m] = j;
                m += 1;
            } else {
                break;
            }
        }
    }
    s->n = m;

    k_sort_frac_descending(s);
}

// (nearly) exhaustive search with cumulative sums
// Need k_sort to have room for n*(max(abs(nmin), abs(nmax))) fractions
static float make_qkxs_quants(int n, const float * GGML_RESTRICT x, const float * GGML_RESTRICT weights, int8_t * GGML_RESTRICT L, int8_t * GGML_RESTRICT Laux, struct k_sort * k_sort, bool signed_scale, bool fast) {
    const int nmin = MIN(0, -k_sort->mid_k); // TODO: maybe pass these directly instead
    const int nmax = MAX(0, k_sort->k + nmin - 1);
    float amax = fabsf(x[0]);
    float w_amax = weights[0] * amax;
    int amax_i = 0;
    int w_amax_i = 0;
    for (int i = 1; i < n; ++i) {
        const float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; amax_i = i; }
        // Find the most important value
        // TODO: use some other metric
        const float w = weights[i];
        const float wax = w * ax;
        if (wax > w_amax) { w_amax = wax; w_amax_i = i; }
    }

    // TODO: less arbitrary threshold
    if (amax < GROUP_MAX_EPS) { // all zero
        memset(L, 0, n);
        return 0.0f;
    }

    bool negative_scale = false;
    if (signed_scale && -nmin != nmax) {
        // the max side should have the biggest range
        // NOTE: this is not perfect for all distributions of weights and very asymmetric quantization.
        if ((x[amax_i] < 0.0f) == (-nmin < nmax)) {
            negative_scale = true;
        }
    }

    // Find the max range in [0, amax_range] which doesn't result in clamping.
    // This could be the range from the side which would clamp first (biggest ratio of max to nmax).
    // But it's simpler to simply pick the smallest side
    int amax_range = MIN(-nmin, nmax);
    if (amax_range == 0) {
        // There's a side which will always clamp anyway
        amax_range = MAX(-nmin, nmax);
    }

    float sumlx = 0.0f;
    float suml2 = 0.0f;
    float best = 0.0f;
    float best_numer = 0.0f;
    float best_denom = 1.0f;
    if (amax_range > 1) {
        // The smallest non-redundant iscale makes the max half+1 its max integer value.
        // Proof: anything smaller has a representable vector with values 2^n as big.
        // Using half (in the slow case) instead of half+1 to avoid skipping some vectors due to rounding imprecisions
        const float iscale = ((float) (fast ? amax_range - 1 : amax_range >> 1)) / amax * (negative_scale ? -1.0f : 1.0f);
        for (int i = 0; i < n; ++i) {
            const float w = weights[i];
            const int l = MAX(nmin, MIN(nearest_int(x[i] * iscale), nmax));
            Laux[i] = l + k_sort->mid_k;
            suml2 += w * l * l;
            sumlx += w * l * x[i];
        }
        if (suml2 > 0.0f) {
            best = sumlx * sumlx;
            best_numer = sumlx;
            best_denom = suml2; // should never be zero
        }
        if (negative_scale) {
            // to let the cumulative sum consider only the positive case,
            // keeping it closer to branch-free
            sumlx = -sumlx;
        }
    } else {
        memset(Laux, k_sort->mid_k, n);
    }
    memcpy(L, Laux, n);

    // Sort the next scales to cumulatively compare them
    k_sort_set_x_L(k_sort, n, w_amax_i, x, Laux, negative_scale);

    int best_p_i = -1; // consecutive with 0..k_sort->n
    for (int i = 0; i < k_sort->n; ++i) {
        // maximize the weighted cosine similarity
        const int ii = k_sort->ids[i];
        const int k_i = k_sort->k_ids[i];
        const float w = weights[ii];
        sumlx += w * fabsf(x[ii]);
        suml2 += w * k_sort->odd[k_i];
        const float current = sumlx * sumlx;
        Laux[ii] = k_i;
        if (suml2 > 0.0f && current * best_denom > best * suml2) {
            best = current;
            best_numer = negative_scale ? -sumlx : sumlx;
            best_denom = suml2;
            if (i == best_p_i + 1) {
                // reduce copies for consecutive bests
                L[ii] = k_i;
            } else {
                memcpy(L, Laux, n);
            }
            best_p_i = i;
        }
    }

    return best_denom > 0.0f ? best_numer / best_denom : 0.0f;
}

// TODO: remove this, heap sort is slow
// incremental search with cumulative sums
static float make_qkxh_quants(int n, const float * GGML_RESTRICT x, const float * GGML_RESTRICT weights, int8_t * GGML_RESTRICT L, int8_t * GGML_RESTRICT Laux, struct k_heap * GGML_RESTRICT k_heap, bool signed_scale) {
    const int nmin = MIN(0, -k_heap->mid_k); // TODO: maybe directly pass these
    const int nmax = MAX(0, k_heap->k + nmin - 1);
    float amax = fabsf(x[0]);
    float w_amax = (weights ? weights[0] : x[0] * x[0]) * amax;
    int amax_i = 0;
    int w_amax_i = 0;
    for (int i = 1; i < n; ++i) {
        // Find the most important value
        const float w = weights ? weights[i] : x[i] * x[i];
        const float ax = fabsf(x[i]);
        const float wax = w * ax;
        if (ax > amax) {
            amax = ax;
            amax_i = i;
        }
        if (wax > w_amax) {
            w_amax = wax;
            w_amax_i = i;
        }
    }

    if (amax < GROUP_MAX_EPS) { // all zero
        memset(L, 0, n);
        return 0.0f;
    }

    bool negative_scale = false;
    if (signed_scale && -nmin > nmax) {
        // the max side should have the biggest range
        // NOTE: this is not always the best sign, but seems to be a good heuristic.
        if ((x[amax_i] < 0.0f) == (-nmin < nmax)) {
            // [-4, 3] ==> [-3, 4]
            negative_scale = true;
        }
    }

    // Find the max range in [0, amax_range] which doesn't result in clamping.
    // This is the range from the side which would clamp first (biggest ratio of max to nmax).
    // But it's easier and safer to simply use the smallest range.
    int amax_range = MIN(-nmin, nmax);
    if (amax_range == 0) {
        // one side will clamp anyway
        amax_range = MAX(-nmin, nmax);
    }
    float sumlx = 0.0f;
    float suml2 = 0.0f;
    float scale = 0.0f;
    float best = 0.0f;
    float best_denom = 1.0f; // should never be zero
    if (amax_range > 1) {
        // The smallest non-redundant iscale makes the first clamped value half+1 its max integer value.
        // Proof: anything smaller has a representable vector with values twice as big.
        // But using amax_range - 1 works very well at least with slightly asymmetric quants (e.g. [-8, 7]),
        // even though it's less exhaustive.
        const float iscale = ((float)(amax_range - 1))/amax * (negative_scale ? -1.0f : 1.0f);
        // const float iscale = ((float)(amax_range >> 2))/amax * (negative_scale ? -1.0f : 1.0f);
        for (int i = 0; i < n; ++i) {
            const float w = weights ? weights[i] : x[i] * x[i];
            int l = MAX(nmin, MIN(nearest_int(x[i] * iscale), nmax));
            Laux[i] = l + k_heap->mid_k;
            suml2 += w * l * l;
            sumlx += w * l * x[i];
        }
        if (suml2 > 0.0f) {
            best = sumlx * sumlx;
            best_denom = suml2;
            scale = sumlx / suml2;
        }
    } else {
        memset(Laux, k_heap->mid_k, n);
    }
    memcpy(L, Laux, n);

    k_heap_set_x_L(k_heap, x, Laux, n, negative_scale);

    const int imax_range = MAX(abs(nmin), abs(nmax));
    const int max_odd = 2*(imax_range + 1) + 1;
    const float wmax = fabsf(x[w_amax_i]);
    int best_p_i = -1; // consecutive with 0..n_frac
    for (int i = 0; k_heap->n > 0; ++i) {
        struct fraction frac = k_heap_pop(k_heap);
        if (frac.numer == 0.0f) { break; }
        const float v_max_odd = frac.numer * max_odd;
        if (wmax * frac.denom > v_max_odd) {
            // stop when the inverse scale would result in clamping the most important value
            break;
        }
        // maximize the weighted cosine similarity
        const int ii = frac.i;
        const float w = weights ? weights[ii] : x[ii] * x[ii];
        if (negative_scale) {
            frac.numer = -frac.numer;
        }
        sumlx += w * frac.numer;
        suml2 += w * frac.denom;
        const float current = sumlx * sumlx;
        Laux[ii] += (x[ii] < 0.0f) != negative_scale ? -1 : 1;
        if (suml2 > 0.0f && current * best_denom > best * suml2) {
            best = current;
            best_denom = suml2;
            scale = sumlx / suml2;
            if (i == best_p_i + 1) {
                // reduce copies for consecutive bests
                L[ii] += (x[ii] < 0.0f) != negative_scale ? -1 : 1;
            } else {
                memcpy(L, Laux, n);
            }
            best_p_i = i;
        }
    }

    return scale;
}

// TODO: remove this, heap sort is slow
// like make_qkxh_quants, but doesn't assume the sign of the scale is the sign of the absmax value
static float make_qkxsh_quants(int n, int nmin, int nmax, const float * GGML_RESTRICT x, const float * GGML_RESTRICT weights, int8_t * GGML_RESTRICT L, int8_t * GGML_RESTRICT Laux, struct k_heap * GGML_RESTRICT k_heap) {
    nmin = MIN(0, nmin);
    nmax = MAX(0, nmax);
    // start at zero
    float amax = fabsf(x[0]);
    float min = x[0];
    float max = x[0];
    float w_amax = weights[0] * amax;
    int w_amax_i = 0;
    for (int i = 1; i < n; ++i) {
        const float w = weights[i];
        const float ax = fabsf(x[i]);
        const float wax = w * ax;
        if (ax > amax) { amax = ax; }
        if (x[i] > max) { max = x[i]; }
        if (x[i] < min) { min = x[i]; }
        // Find the most important value
        if (wax > w_amax) { w_amax = wax; w_amax_i = i; }
    }

    if (amax < GROUP_MAX_EPS) { // all zero
        memset(L, 0, n);
        return 0.0f;
    }

    // Use the side which will clamp first.
    // The first clamped value is the absmax at the end of the common range.
    int amax_range = MIN(-nmin, nmax);
    if (amax_range == 0) {
        // One side will always clamp anyway
        amax_range = MAX(-nmin, nmax);
    }
    float sumlx_p = 0.0f;
    float suml2_p = 0.0f;
    float sumlx_n = 0.0f;
    float suml2_n = 0.0f;
    float scale = 0.0f;
    float best = 0.0f;
    float best_denom = 1.0f;
    int best_i = -1; // consecutive with 0..n_frac
    // Pre-calculate the half-point for the common range.
    // All smaller vectors have a representable vector with twice the values, and thus can be skipped.
    // But using amax_range - 1 works very well at least with slightly asymmetric quants (e.g. [-8, 7]),
    // even though it's less exhaustive.
    if (amax_range > 1) {
        const float iscale = ((float)(amax_range - 1))/amax;
        for (int i = 0; i < n; ++i) {
            const float w = weights[i];
            int l = MAX(nmin, MIN(nearest_int(x[i] * iscale), nmax));
            Laux[i] = l + k_heap->mid_k;
            suml2_p += w * l * l;
            sumlx_p += w * l * x[i];
        }
        sumlx_n = -sumlx_p;
        suml2_n = suml2_p;
        const float current_p = sumlx_p * sumlx_p;
        if (suml2_p > 0.0f) {
            best = current_p;
            best_denom = suml2_p;
            scale = sumlx_p / suml2_p;
            best_i = -1; // right before 0 of the loop after sorting
        }
    } else {
        memset(Laux, k_heap->mid_k, n);
    }
    memcpy(L, Laux, n);

    k_heap_set_x_L(k_heap, x, Laux, n, false);

    // TODO: make that range sign-aware to reduce the search space
    const int imax_range = MAX(nmax, -nmin);
    const int max_odd = 2*(imax_range + 1) + 1;
    const float wmax = fabsf(x[w_amax_i]);

    const float max_common_odd = (MIN(nmax, -nmin) * 2) + 1;
    const float max_odd_p = (nmax * 2) + 1;
    const float max_odd_n = (-nmin * 2) + 1;
    for (int i = 0; k_heap->n > 0; ++i) {
        struct fraction frac = k_heap_pop(k_heap);
        // maximize the weighted cosine similarity
        const int ii = frac.i;
        const float w = weights[ii];
        const float lx = w * frac.numer;
        const float odd = frac.denom;
        const float l2 = w * odd;
        if (wmax * odd > frac.numer * max_odd) {
            // stop when the inverse scale would result in clamping the most important value
            break;
        }

        Laux[ii] += x[ii] < 0.0f ? -1 : 1;

        float sumlx;
        float proj;
        float norm;
        if (odd < max_common_odd) {
            sumlx_p += lx;
            suml2_p += l2;
            sumlx_n -= lx;
            suml2_n += l2;

            sumlx = sumlx_p;
            proj = sumlx_p * sumlx_p;
            norm = suml2_p;

            // avoid double-copying Laux in a single iteration
            if (suml2_p != suml2_n && suml2_p * suml2_n > 0.0f) {
                const float proj_n = sumlx_n * sumlx_n;
                if (proj_n * norm > proj * suml2_n) {
                    sumlx = sumlx_n;
                    proj = proj_n;
                    norm = suml2_n;
                }
            }
        } else if (x[ii] < 0.0f ? odd < max_odd_n : odd < max_odd_p) {
            sumlx_p += lx;
            suml2_p += l2;

            sumlx = sumlx_p;
            proj = sumlx_p * sumlx_p;
            norm = suml2_p;
        } else {
            // outside the positive range means we're now into negatives
            sumlx_n -= lx;
            suml2_n += l2;

            sumlx = sumlx_n;
            proj = sumlx_n * sumlx_n;
            norm = suml2_n;
        }
        if (norm > 0.0f && proj * best_denom > best * norm) {
            best = proj;
            best_denom = norm;
            scale = sumlx / norm;
            if (i == best_i + 1) {
                // reduce copies for consecutive bests
                L[ii] += x[ii] < 0.0f ? -1 : 1;
            } else {
                memcpy(L, Laux, n);
            }
            best_i = i;
        }
    }

    if (scale < 0.0f) {
        for (int i = 0; i < n; ++i) {
            L[i] = MAX(nmin, MIN(-(L[i] - k_heap->mid_k), nmax)) - nmin;
        }
    } else {
        for (int i = 0; i < n; ++i) {
            L[i] = MAX(nmin, MIN(L[i] - k_heap->mid_k, nmax)) - nmin;
        }
    }

    return scale;
}

// TODO: remove this, heap sort is slow
// fast cumulative search for non-linear quantization
static float make_qkxh_nl_fast_quants(int n, const float * GGML_RESTRICT x, const float * GGML_RESTRICT weights, int8_t * GGML_RESTRICT L, int8_t * GGML_RESTRICT Laux, struct k_heap * GGML_RESTRICT k_heap, bool signed_scale) {
    float sumlx = 0.0f;
    float suml2 = 0.0f;
    float amax = -1.0f;
    float w_amax = -1.0f;
    int amax_i = -1;
    int w_amax_i = -1;
    for (int i = 0; i < n; ++i) {
        const float w = weights ? weights[i] : x[i] * x[i];
        const float ax = fabsf(x[i]);
        const float wax = w * ax;
        if (ax > amax) {
            amax = ax;
            amax_i = i;
        }
        if (wax > w_amax) {
            w_amax = wax;
            w_amax_i = i;
        }
    }

    if (amax < GROUP_MAX_EPS) { // all zero
        memset(L, 0, n);
        return 0.0f;
    }

    const bool neg_scale = signed_scale ? (x[amax_i] < 0.0f) != (k_heap->kmax < 0) : false;

    // Start searching with the max at the second last step
    const int amax_range = abs(k_heap->kmax) - k_heap->kmax_step;
    // Other starting ranges can be used too.
    // const int amax_range = abs(k_heap->kvalues[0]) > abs(k_heap->kvalues[k_heap->k - 1]) ?
    //                            abs(k_heap->kvalues[k_heap->k - 2]) :
    //                            abs(k_heap->kvalues[1]);
    const float iscale = amax_range / amax * (neg_scale ? -1.0f : 1.0f);
    for (int i = 0; i < n; ++i) {
        const float w = weights ? weights[i] : x[i] * x[i];
        int l = k_heap_best_index(k_heap, x[i] * iscale);
        int q = (int) k_heap->kvalues[l];
        Laux[i] = l;
        sumlx += w * q * x[i];
        suml2 += w * q * q;
    }
    memcpy(L, Laux, n);

    k_heap_set_x_L(k_heap, x, Laux, n, neg_scale);

    float best;
    float best_sumlx;
    float best_suml2;
    if (suml2 != 0.0f) {
        best = sumlx * sumlx;
        best_sumlx = sumlx; // can't change the sign of kmin
        best_suml2 = suml2;
    } else {
        best = 0.0f;
        best_sumlx = 0.0f;
        best_suml2 = 1.0f;
    }
    const int max_odd = 2 * abs(k_heap->kmax) + (int) k_heap->kmax_step;
    const float wmax = fabsf(x[w_amax_i]);
    float sumlx_p = neg_scale ? -sumlx : sumlx;
    float suml2_p = suml2;
    int best_p_i = -1; // consecutive with 0..n_frac
    for (int i = 0; k_heap->n > 0; ++i) {
        struct fraction frac = k_heap_pop(k_heap);
        const int ii = frac.i;
        const float w = weights ? weights[ii] : x[ii] * x[ii];
        const float v_max_odd = frac.numer * max_odd;
        if (wmax * frac.denom > v_max_odd) {
            // stop when the inverse scale would result in clamping the most important value
            break;
        }
        sumlx_p += w * frac.numer;
        suml2_p += w * frac.denom;
        const float current = sumlx_p * sumlx_p;
        Laux[ii] += (x[ii] < 0.0f) != neg_scale ? -1 : 1;
        if (suml2_p > 0.0f && current * best_suml2 > best * suml2_p) {
            best = current;
            best_sumlx = neg_scale ? -sumlx_p : sumlx_p;
            best_suml2 = suml2_p;
            if (i == best_p_i + 1) {
                // reduce copies for consecutive bests
                L[ii] += (x[ii] < 0.0f) != neg_scale ? -1 : 1;
            } else {
                memcpy(L, Laux, n);
            }
            best_p_i = i;
        }
    }

    return best_suml2 > 0.0f ? best_sumlx / best_suml2 : 0.0f;
}

// TODO: remove this, heap sort is slow
// non-linear (nearly) exhaustive search with cumulative sums
static float make_qkxh_nl_quants(int n, const float * GGML_RESTRICT x, const float * GGML_RESTRICT weights, uint8_t * GGML_RESTRICT L, uint8_t * GGML_RESTRICT Laux, struct k_heap * GGML_RESTRICT k_heap, bool signed_scale, bool fast) {
    float sumlx = 0.0f;
    float suml2 = 0.0f;
    float amax = -1.0f;
    int amax_i = -1;
    const int8_t kmin = k_heap->kmin;
    for (int i = 0; i < n; ++i) {
        const float w = weights ? weights[i] : x[i] * x[i];
        const float ax = fabsf(x[i]);
        if (ax > amax) {
            amax = ax;
            amax_i = i;
        }
        sumlx += w * x[i] * kmin;
        suml2 += w * kmin * kmin;
    }
    memset(Laux, k_heap->mid_k, n);
    memset(L, k_heap->mid_k, n);

    const bool neg_scale = signed_scale && fast ? (x[amax_i] < 0.0f) != (k_heap->kmax < 0) : false;

    k_heap_set_x(k_heap, x, n, neg_scale);

    float best;
    float best_sumlx;
    float best_suml2;
    if (suml2 != 0.0f) {
        best = sumlx * sumlx;
        best_sumlx = sumlx; // can't change the sign of kmin
        best_suml2 = suml2;
    } else {
        best = 0.0f;
        best_sumlx = 0.0f;
        best_suml2 = 1.0f;
    }
    float sumlx_p = neg_scale ? -sumlx : sumlx;
    float suml2_p = suml2;
    int best_p_i = -1; // consecutive with 0..n_frac
    for (int i = 0; k_heap->n > 0; ++i) {
        struct fraction frac = k_heap_pop(k_heap);
        const int ii = frac.i;
        const float w = weights ? weights[ii] : x[ii] * x[ii];
        sumlx_p += w * frac.numer;
        suml2_p += w * frac.denom;
        const float current = sumlx_p * sumlx_p;
        Laux[ii] += (x[ii] < 0.0f) != neg_scale ? -1 : 1;
        if (suml2_p > 0.0f && current * best_suml2 > best * suml2_p) {
            best = current;
            best_sumlx = neg_scale ? -sumlx_p : sumlx_p;
            best_suml2 = suml2_p;
            if (i == best_p_i + 1) {
                // reduce copies for consecutive bests
                L[ii] += (x[ii] < 0.0f) != neg_scale ? -1 : 1;
            } else {
                memcpy(L, Laux, n);
            }
            best_p_i = i;
        }
    }

    // Non-linear mappings are usually not symmetric, so try negating the scale
    // This is the same as above, but keeping the old best if the new best is not better.
    if (signed_scale && !fast) {
        memset(Laux, k_heap->mid_k, n);

        k_heap_set_x(k_heap, x, n, true);

        float sumlx_n = -sumlx;
        float suml2_n = suml2;
        int best_n_i = -2; // not consecutive with 0..n_frac
        for (int i = 0; k_heap->n > 0; ++i) {
            struct fraction frac = k_heap_pop(k_heap);
            const int ii = frac.i;
            const float w = weights ? weights[ii] : x[ii] * x[ii];
            sumlx_n += w * frac.numer;
            suml2_n += w * frac.denom;
            const float current = sumlx_n * sumlx_n;
            Laux[ii] += x[ii] >= 0.0f ? -1 : 1;
            if (suml2_n > 0.0f && current * best_suml2 > best * suml2_n) {
                best = current;
                best_sumlx = -sumlx_n;
                best_suml2 = suml2_n;
                if (i == best_n_i + 1) {
                    // reduce copies for consecutive bests
                    L[ii] += x[ii] >= 0.0f ? -1 : 1;
                } else {
                    memcpy(L, Laux, n);
                }
                best_n_i = i;
            }
        }
    }

    return best_suml2 != 0.0f ? best_sumlx / best_suml2 : 0.0f;
}

// non-linear (nearly) exhaustive search with cumulative sums
static float make_qkxs_nl_quants(int n, const float * GGML_RESTRICT x, const float * GGML_RESTRICT weights, int8_t * GGML_RESTRICT L, int8_t * GGML_RESTRICT Laux, struct k_sort * GGML_RESTRICT k_sort, bool signed_scale, bool fast) {
    float sumlx = 0.0f;
    float suml2 = 0.0f;
    float amax = -1.0f;
    float w_amax = -1.0f;
    int amax_i = -1;
    int w_amax_i = -1;
    const int8_t kmin = k_sort->kmin;
    for (int i = 0; i < n; ++i) {
        const float w = weights ? weights[i] : x[i] * x[i];
        const float ax = fabsf(x[i]);
        const float wax = w * ax;
        if (ax > amax) {
            amax = ax;
            amax_i = i;
        }
        if (wax > w_amax) {
            w_amax = wax;
            w_amax_i = i;
        }
        sumlx += w * x[i] * kmin;
        suml2 += w * kmin * kmin;
    }

    if (amax < GROUP_MAX_EPS) {  // all zero
        memset(L, 0, n);
        return 0.0f;
    }

    memset(Laux, k_sort->mid_k, n);
    memset(L, k_sort->mid_k, n);

    const bool neg_scale = signed_scale && fast ? (x[amax_i] < 0.0f) != (k_sort->kmax < 0) : false;

    k_sort_set_x_L(k_sort, n, w_amax_i, x, Laux, neg_scale);

    float best;
    float best_sumlx;
    float best_suml2;
    if (suml2 != 0.0f) {
        best = sumlx * sumlx;
        best_sumlx = sumlx; // can't change the sign of kmin
        best_suml2 = suml2;
    } else {
        best = 0.0f;
        best_sumlx = 0.0f;
        best_suml2 = 1.0f;
    }
    float sumlx_p = neg_scale ? -sumlx : sumlx;
    float suml2_p = suml2;
    int best_p_i = -1; // consecutive with 0..k_sort->n
    for (int i = 0; i < k_sort->n; ++i) {
        const int ii = k_sort->ids[i];
        const int k_i = k_sort->k_ids[i];
        const float odd = k_sort->odd[k_i];
        const float step = k_sort->step[k_i];
        const float w = weights ? weights[ii] : x[ii] * x[ii];
        sumlx_p += w * (fabsf(x[ii]) * step);
        suml2_p += w * (odd * step);
        const float current = sumlx_p * sumlx_p;
        Laux[ii] = k_i;
        if (suml2_p > 0.0f && current * best_suml2 > best * suml2_p) {
            best = current;
            best_sumlx = neg_scale ? -sumlx_p : sumlx_p;
            best_suml2 = suml2_p;
            if (i == best_p_i + 1) {
                // reduce copies for consecutive bests
                L[ii] = k_i;
            } else {
                memcpy(L, Laux, n);
            }
            best_p_i = i;
        }
    }

    // Non-linear mappings are usually not symmetric, so try negating the scale
    // This is the same as above, but keeping the old best if the new best is not better.
    if (signed_scale && !fast) {
        memset(Laux, k_sort->mid_k, n);

        k_sort_set_x_L(k_sort, n, w_amax_i, x, Laux, true);

        float sumlx_n = -sumlx;
        float suml2_n = suml2;
        int best_n_i = -2; // not consecutive with 0..k_sort->n
        for (int i = 0; i < k_sort->n; ++i) {
            const int ii = k_sort->ids[i];
            const int k_i = k_sort->k_ids[i];
            const float odd = k_sort->odd[k_i];
            const float step = k_sort->step[k_i];
            const float w = weights ? weights[ii] : x[ii] * x[ii];
            sumlx_n += w * (fabsf(x[ii]) * step);
            suml2_n += w * (odd * step);
            const float current = sumlx_n * sumlx_n;
            Laux[ii] = k_i;
            if (suml2_n > 0.0f && current * best_suml2 > best * suml2_n) {
                best = current;
                best_sumlx = -sumlx_n;
                best_suml2 = suml2_n;
                if (i == best_n_i + 1) {
                    // reduce copies for consecutive bests
                    L[ii] = k_i;
                } else {
                    memcpy(L, Laux, n);
                }
                best_n_i = i;
            }
        }
    }

    return best_suml2 != 0.0f ? best_sumlx / best_suml2 : 0.0f;
}

static inline void get_scale_min_k4(int j, const uint8_t * GGML_RESTRICT q, uint8_t * GGML_RESTRICT d, uint8_t * GGML_RESTRICT m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

//========================- 2-bit (de)-quantization

void quantize_row_q2_K_ref(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[16];
    float   weights[16];
    float mins[QK_K/16];
    float scales[QK_K/16];

    const float q4scale = 15.f;

    for (int i = 0; i < nb; i++) {
        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 16; ++l) weights[l] = fabsf(x[16*j + l]);
            scales[j] = make_qkx2_quants(16, 3, x + 16*j, weights, L + 16*j, &mins[j], Laux, -0.5f, 0.1f, 15, true);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

        if (max_scale > 0) {
            float iscale = q4scale/max_scale;
            for (int j = 0; j < QK_K/16; ++j) {
                int l = nearest_int(iscale*scales[j]);
                y[i].scales[j] = l;
            }
            y[i].d = GGML_FP32_TO_FP16(max_scale/q4scale);
        } else {
            for (int j = 0; j < QK_K/16; ++j) y[i].scales[j] = 0;
            y[i].d = GGML_FP32_TO_FP16(0.f);
        }
        if (max_min > 0) {
            float iscale = q4scale/max_min;
            for (int j = 0; j < QK_K/16; ++j) {
                int l = nearest_int(iscale*mins[j]);
                y[i].scales[j] |= (l << 4);
            }
            y[i].dmin = GGML_FP32_TO_FP16(max_min/q4scale);
        } else {
            y[i].dmin = GGML_FP32_TO_FP16(0.f);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            const float d = GGML_FP16_TO_FP32(y[i].d) * (y[i].scales[j] & 0xF);
            if (!d) continue;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * (y[i].scales[j] >> 4);
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int((x[16*j + ii] + dm)/d);
                l = MAX(0, MIN(3, l));
                L[16*j + ii] = l;
            }
        }

        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[i].qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }

        x += QK_K;
    }
}

void dequantize_row_q2_K(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);

        const uint8_t * q = x[i].qs;

        int is = 0;
        float dl, ml;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {

                uint8_t sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;

                sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml;

                shift += 2;
            }
            q += 32;
        }
    }
}

static float make_qkx3_quants(int n, int nmax, const float * GGML_RESTRICT x, const float * GGML_RESTRICT weights,
        uint8_t * GGML_RESTRICT L, float * GGML_RESTRICT the_min, uint8_t * GGML_RESTRICT Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights ? weights[0] : x[0]*x[0];
    float sum_x = sum_w * x[0];
#ifdef HAVE_BUGGY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
    for (volatile int i = 1; i < n; ++i) {
#else
    for (int i = 1; i < n; ++i) {
#endif
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights ? weights[i] : x[i]*x[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) {
        min = 0;
    }
    if (max <= min) {
        memset(L, 0, n);
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    float best_mad = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale*(x[i] - min));
        L[i] = MAX(0, MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff*diff;
        float w = weights ? weights[i] : x[i]*x[i];
        best_mad += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = MAX(0, MIN(nmax, l));
            Laux[i] = l;
            float w = weights ? weights[i] : x[i]*x[i];
            sum_l  += w*l;
            sum_l2 += w*l*l;
            sum_xl += w*l*x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float mad = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff*diff;
                float w = weights ? weights[i] : x[i]*x[i];
                mad += w * diff;
            }
            if (mad < best_mad) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

static float make_qp_quants(int n, int nmax, const float * GGML_RESTRICT x, uint8_t * GGML_RESTRICT L, const float * quant_weights) {
    float max = 0;
    for (int i = 0; i < n; ++i) {
        max = MAX(max, x[i]);
    }
    if (!max) { // all zero
        for (int i = 0; i < n; ++i) { L[i] = 0; }
        return 0.f;
    }
    float iscale = nmax / max;
    for (int i = 0; i < n; ++i) {
        L[i] = nearest_int(iscale * x[i]);
    }
    float scale = 1/iscale;
    float best_mse = 0;
    for (int i = 0; i < n; ++i) {
        float diff = x[i] - scale*L[i];
        float w = quant_weights[i];
        best_mse += w*diff*diff;
    }
    for (int is = -4; is <= 4; ++is) {
        if (is == 0) continue;
        float iscale_is = (0.1f*is + nmax)/max;
        float scale_is = 1/iscale_is;
        float mse = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale_is*x[i]);
            l = MIN(nmax, l);
            float diff = x[i] - scale_is*l;
            float w = quant_weights[i];
            mse += w*diff*diff;
        }
        if (mse < best_mse) {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = MIN(nmax, l);
        L[i] = l;
        float w = quant_weights[i];
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int i = 0; i < n; ++i) {
            float w = quant_weights[i];
            float slx = sumlx - w*x[i]*L[i];
            float sl2 = suml2 - w*L[i]*L[i];
            if (slx > 0 && sl2 > 0) {
                int new_l = nearest_int(x[i] * sl2 / slx);
                new_l = MIN(nmax, new_l);
                if (new_l != L[i]) {
                    slx += w*x[i]*new_l;
                    sl2 += w*new_l*new_l;
                    if (slx*slx*suml2 > sumlx*sumlx*sl2) {
                        L[i] = new_l; sumlx = slx; suml2 = sl2;
                        ++n_changed;
                    }
                }
            }
        }
        if (!n_changed) {
            break;
        }
    }
    return sumlx/suml2;
}

static void quantize_row_q2_K_impl(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int k, const float * GGML_RESTRICT quant_weights) {
    GGML_ASSERT(quant_weights);
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    const bool requantize = true;

    uint8_t L[QK_K];
    uint8_t Laux[16];
    float mins[QK_K/16];
    float scales[QK_K/16];
    float sw[QK_K/16];
    float weight[16];
    uint8_t Ls[QK_K/16], Lm[QK_K/16];

    struct k_sort k_sort;
    uint8_t k_buf[K_SORT_BUF_SIZE_LINEAR(QK_K/16, 0, 15)];

    k_sort_init_linear(&k_sort, QK_K/16, 0, 15, k_buf);

    for (int i = 0; i < nb; i++) {
        memset(sw, 0, QK_K/16*sizeof(float));
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += x[j]*x[j];
        float sigma2 = sumx2/QK_K;
        for (int j = 0; j < QK_K/16; ++j) {
            const float * GGML_RESTRICT qw = quant_weights + QK_K * i + 16*j;
            for (int l = 0; l < 16; ++l) weight[l] = qw[l] * sqrtf(sigma2 + x[16*j + l]*x[16*j + l]);
            for (int l = 0; l < QK_K/16; ++l) sw[j] += weight[l];
            scales[j] = make_qkx3_quants(16, 3, x + 16*j, weight, L + 16*j, &mins[j], Laux, -0.9f, 0.05f, 36, false);
        }

        float dm, mm;
        dm  = make_qkxs_quants(QK_K/16, scales, sw, (int8_t *) Ls, (int8_t *) Laux, &k_sort, false, true);
        mm  = make_qkxs_quants(QK_K/16, mins, sw, (int8_t *) Lm, (int8_t *) Laux, &k_sort, false, true);
        // dm  = make_qp_quants(QK_K/16, 15, scales, Ls, sw);
        // mm  = make_qp_quants(QK_K/16, 15, mins,   Lm, sw);

        y[i].d    = GGML_FP32_TO_FP16(dm);
        y[i].dmin = GGML_FP32_TO_FP16(mm);
        dm        = GGML_FP16_TO_FP32(y[i].d);
        mm        = GGML_FP16_TO_FP32(y[i].dmin);

        for (int j = 0; j < QK_K/16; ++j) {
            y[i].scales[j] = Ls[j] | (Lm[j] << 4);
        }

        // TODO: avoid requant
        if (requantize) {
            for (int j = 0; j < QK_K/16; ++j) {
                const float d = dm * (y[i].scales[j] & 0xF);
                if (!d) continue;
                const float m = mm * (y[i].scales[j] >> 4);
                for (int ii = 0; ii < 16; ++ii) {
                    int l = nearest_int((x[16*j + ii] + m)/d);
                    l = MAX(0, MIN(3, l));
                    L[16*j + ii] = l;
                }
            }
        }

        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[i].qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }

        x += QK_K;
    }
}

size_t quantize_q2_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    size_t row_size = ggml_row_size(GGML_TYPE_Q2_K, n_per_row);
    if (!quant_weights) {
        quantize_row_q2_K_ref(src, dst, (int64_t)nrow*n_per_row);
    }
    else {
        char * qrow = (char *)dst;
        for (int64_t row = 0; row < nrow; ++row) {
            quantize_row_q2_K_impl(src, (block_q2_K*)qrow, n_per_row, quant_weights);
            src += n_per_row;
            qrow += row_size;
        }
    }
    return nrow * row_size;
}

//========================= 3-bit (de)-quantization

void quantize_row_q3_K_ref(const float * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    int8_t L[QK_K];
    int8_t Laux[16];
    float scales[QK_K / 16];
    float weights[16];
    struct k_sort k_sort;
    uint8_t k_buf[K_SORT_BUF_SIZE_LINEAR(16, -4, 3)];

    k_sort_init_linear(&k_sort, 16, -4, 3, k_buf);

    for (int i = 0; i < 16; ++i) {
        weights[i] = 1.0f;
    }

    for (int i = 0; i < nb; i++) {

        float max_scale = 0;
        float amax = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            scales[j] = make_qkxs_quants(16, x + 16*j, weights, L + 16*j, Laux, &k_sort, true, true);
            float scale = fabsf(scales[j]);
            if (scale > amax) {
                amax = scale; max_scale = scales[j];
            }
        }

        memset(y[i].scales, 0, 12);
        if (max_scale) {
            float iscale = -32.f/max_scale;
            for (int j = 0; j < QK_K/16; ++j) {
                int8_t l = nearest_int(iscale*scales[j]);
                l = MAX(-32, MIN(31, l)) + 32;
                if (j < 8) {
                    y[i].scales[j] = l & 0xF;
                } else {
                    y[i].scales[j-8] |= ((l & 0xF) << 4);
                }
                l >>= 4;
                y[i].scales[j%4 + 8] |= (l << (2*(j/4)));
            }
            y[i].d = GGML_FP32_TO_FP16(1/iscale);
        } else {
            y[i].d = GGML_FP32_TO_FP16(0.f);
        }

        memset(y[i].hmask, 0, QK_K/8);
        // We put the high-bit for the 1st 8 quants into bit 0, the next 8 into bit 1, etc.
        int m = 0;
        uint8_t hm = 1;
        for (int j = 0; j < QK_K; ++j) {
            if (L[j] > 3) {
                y[i].hmask[m] |= hm;
                L[j] -= 4;
            }
            if (++m == QK_K/8) {
                m = 0; hm <<= 1;
            }
        }
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[i].qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }

        x += QK_K;
    }
}

void dequantize_row_q3_K(const block_q3_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    uint32_t aux[4];
    const int8_t * scales = (const int8_t*)aux;

    for (int i = 0; i < nb; i++) {

        const float d_all = GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * GGML_RESTRICT q = x[i].qs;
        const uint8_t * GGML_RESTRICT hm = x[i].hmask;
        uint8_t m = 1;

        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        float dl;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l+ 0] >> shift) & 3) - ((hm[l+ 0] & m) ? 0 : 4));
                }

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4));
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }

    }
}

static void quantize_row_q3_K_impl(const float * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t n_per_row, const float * GGML_RESTRICT quant_weights) {
    assert(n_per_row % QK_K == 0);
    const int nb = n_per_row / QK_K;

    int8_t L[QK_K];
    int8_t Laux[16];
    float scales[QK_K / 16];
    float weight[16];
    float sw[QK_K / 16];
    int8_t Ls[QK_K / 16];
    struct k_sort k_sort;
    struct k_sort k_sort_s;
    uint8_t k_buf[K_SORT_BUF_SIZE_LINEAR(16, -4, 3)];
    uint8_t k_buf_s[K_SORT_BUF_SIZE_LINEAR(QK_K / 16, -32, 31)];

    k_sort_init_linear(&k_sort, 16, -4, 3, k_buf);
    k_sort_init_linear(&k_sort_s, QK_K / 16, -32, 31, k_buf_s);

    for (int i = 0; i < nb; i++) {

        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += x[j]*x[j];
        float sigma2 = 2*sumx2/QK_K;

        for (int j = 0; j < QK_K/16; ++j) {
            if (quant_weights) {
                const float * qw = quant_weights + QK_K * i + 16*j;
                for (int l = 0; l < 16; ++l) weight[l] = qw[l] * sqrtf(sigma2 + x[16*j+l]*x[16*j+l]);
            } else {
                for (int l = 0; l < 16; ++l) weight[l] = x[16*j+l]*x[16*j+l];
            }
            float sumw = 0;
            for (int l = 0; l < 16; ++l) sumw += weight[l];
            sw[j] = sumw;

            scales[j] = make_qkxs_quants(16, x + 16*j, weight, L + 16*j, Laux, &k_sort, true, false);

        }

        memset(y[i].scales, 0, 12);

        float d_block = make_qkxs_quants(QK_K/16, scales, sw, Ls, Laux, &k_sort_s, true, false);
        for (int j = 0; j < QK_K/16; ++j) {
            int l = Ls[j];
            if (j < 8) {
                y[i].scales[j] = l & 0xF;
            } else {
                y[i].scales[j-8] |= ((l & 0xF) << 4);
            }
            l >>= 4;
            y[i].scales[j%4 + 8] |= (l << (2*(j/4)));
        }
        y[i].d = GGML_FP32_TO_FP16(d_block);

        memset(y[i].hmask, 0, QK_K/8);
        // We put the high-bit for the 1st 8 quants into bit 0, the next 8 into bit 1, etc.
        int m = 0;
        uint8_t hm = 1;
        for (int j = 0; j < QK_K; ++j) {
            if (L[j] > 3) {
                y[i].hmask[m] |= hm;
                L[j] -= 4;
            }
            if (++m == QK_K/8) {
                m = 0; hm <<= 1;
            }
        }
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[i].qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }

        x += QK_K;
    }
}

size_t quantize_q3_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    size_t row_size = ggml_row_size(GGML_TYPE_Q3_K, n_per_row);
    if (!quant_weights) {
        quantize_row_q3_K_ref(src, dst, (int64_t)nrow*n_per_row);
    }
    else {
        char * qrow = (char *)dst;
        for (int64_t row = 0; row < nrow; ++row) {
            quantize_row_q3_K_impl(src, (block_q3_K*)qrow, n_per_row, quant_weights);
            src += n_per_row;
            qrow += row_size;
        }
    }
    return nrow * row_size;
}

// ====================== 4-bit (de)-quantization

void quantize_row_q4_K_ref(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[32];
    float   weights[32];
    float mins[QK_K/32];
    float scales[QK_K/32];

    for (int i = 0; i < nb; i++) {
        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            //scales[j] = make_qkx1_quants(32, 15, x + 32*j, L + 32*j, &mins[j], 9, 0.5f);
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) sum_x2 += x[32*j + l] * x[32*j + l];
            float av_x = sqrtf(sum_x2/32);
            for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
            scales[j] = make_qkx2_quants(32, 15, x + 32*j, weights, L + 32*j, &mins[j], Laux, -1.f, 0.1f, 20, false);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

        float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
        float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t ls = nearest_int(inv_scale*scales[j]);
            uint8_t lm = nearest_int(inv_min*mins[j]);
            ls = MIN(63, ls);
            lm = MIN(63, lm);
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } else {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d = GGML_FP32_TO_FP16(max_scale/63.f);
        y[i].dmin = GGML_FP32_TO_FP16(max_min/63.f);

        // TODO: avoid requant
        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            if (!d) continue;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32*j + ii] + dm)/d);
                l = MAX(0, MIN(15, l));
                L[32*j + ii] = l;
            }
        }

        uint8_t * q = y[i].qs;
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; ++l) q[l] = L[j + l] | (L[j + l + 32] << 4);
            q += 32;
        }

        x += QK_K;
    }
}

void dequantize_row_q4_K(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t * q = x[i].qs;

        const float d   = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

static void quantize_row_q4_K_impl(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t n_per_row, const float * quant_weights) {
    assert(n_per_row % QK_K == 0);
    const int64_t nb = n_per_row / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[32];
    uint8_t Ls[QK_K/32];
    uint8_t Lm[QK_K/32];
    float   weights[32];
    float   sw[QK_K/32];
    float   mins[QK_K/32];
    float   scales[QK_K/32];
    struct k_sort k_sort;
    uint8_t k_buf[K_SORT_BUF_SIZE_LINEAR(QK_K/32, 0, 63)];

    k_sort_init_linear(&k_sort, QK_K/32, 0, 63, k_buf);

    for (int i = 0; i < nb; i++) {

        float sum_x2 = 0;
        for (int l = 0; l < QK_K; ++l) sum_x2 += x[l] * x[l];
        float sigma2 = 2*sum_x2/QK_K;
        float av_x = sqrtf(sigma2);

        for (int j = 0; j < QK_K/32; ++j) {
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*i + 32*j;
                for (int l = 0; l < 32; ++l) weights[l] = qw[l] * sqrtf(sigma2 + x[32*j + l]*x[32*j + l]);
            } else {
                for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
            }
            float sumw = 0;
            for (int l = 0; l < 32; ++l) sumw += weights[l];
            sw[j] = sumw;
            scales[j] = make_qkx3_quants(32, 15, x + 32*j, weights, L + 32*j, &mins[j], Laux, -0.9f, 0.05f, 36, false);
        }

        float d_block = make_qkxs_quants(QK_K/32, scales, sw, (int8_t *) Ls, (int8_t *) Laux, &k_sort, false, true);
        float m_block = make_qkxs_quants(QK_K/32, mins,   sw, (int8_t *) Lm, (int8_t *) Laux, &k_sort, false, true);

        // float d_block = make_qp_quants(QK_K/32, 63, scales, Ls, sw);
        // float m_block = make_qp_quants(QK_K/32, 63, mins,   Lm, sw);
        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t ls = Ls[j];
            uint8_t lm = Lm[j];
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } else {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d = GGML_FP32_TO_FP16(d_block);
        y[i].dmin = GGML_FP32_TO_FP16(m_block);

        // TODO: avoid requant
        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            if (!d) continue;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32*j + ii] + dm)/d);
                l = MAX(0, MIN(15, l));
                L[32*j + ii] = l;
            }
        }
        uint8_t * q = y[i].qs;
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; ++l) q[l] = L[j + l] | (L[j + l + 32] << 4);
            q += 32;
        }

        x += QK_K;

    }
}

size_t quantize_q4_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    size_t row_size = ggml_row_size(GGML_TYPE_Q4_K, n_per_row);
    if (!quant_weights) {
        quantize_row_q4_K_ref(src, dst, (int64_t)nrow*n_per_row);
    }
    else {
        char * qrow = (char *)dst;
        for (int64_t row = 0; row < nrow; ++row) {
            quantize_row_q4_K_impl(src, (block_q4_K*)qrow, n_per_row, quant_weights);
            src += n_per_row;
            qrow += row_size;
        }
    }
    return nrow * row_size;
}

// ====================== 5-bit (de)-quantization

void quantize_row_q5_K_ref(const float * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    uint8_t L[QK_K];
    float mins[QK_K/32];
    float scales[QK_K/32];
    float weights[32];
    uint8_t Laux[32];

    for (int i = 0; i < nb; i++) {
        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            //scales[j] = make_qkx1_quants(32, 31, x + 32*j, L + 32*j, &mins[j], 9, 0.5f);
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) sum_x2 += x[32*j + l] * x[32*j + l];
            float av_x = sqrtf(sum_x2/32);
            for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
            scales[j] = make_qkx2_quants(32, 31, x + 32*j, weights, L + 32*j, &mins[j], Laux, -0.5f, 0.1f, 15, false);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

        float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
        float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t ls = nearest_int(inv_scale*scales[j]);
            uint8_t lm = nearest_int(inv_min*mins[j]);
            ls = MIN(63, ls);
            lm = MIN(63, lm);
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } else {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d = GGML_FP32_TO_FP16(max_scale/63.f);
        y[i].dmin = GGML_FP32_TO_FP16(max_min/63.f);

        // TODO: avoid requant
        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            if (!d) continue;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32*j + ii] + dm)/d);
                l = MAX(0, MIN(31, l));
                L[32*j + ii] = l;
            }
        }

        uint8_t * GGML_RESTRICT qh = y[i].qh;
        uint8_t * GGML_RESTRICT ql = y[i].qs;
        memset(qh, 0, QK_K/8);

        uint8_t m1 = 1, m2 = 2;
        for (int n = 0; n < QK_K; n += 64) {
            for (int j = 0; j < 32; ++j) {
                int l1 = L[n + j];
                if (l1 > 15) {
                    l1 -= 16; qh[j] |= m1;
                }
                int l2 = L[n + j + 32];
                if (l2 > 15) {
                    l2 -= 16; qh[j] |= m2;
                }
                ql[j] = l1 | (l2 << 4);
            }
            m1 <<= 2; m2 <<= 2;
            ql += 32;
        }

        x += QK_K;
    }
}

void dequantize_row_q5_K(const block_q5_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t * ql = x[i].qs;
        const uint8_t * qh = x[i].qh;

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

static void quantize_row_q5_K_impl(const float * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t n_per_row, const float * quant_weights) {
    assert(n_per_row % QK_K == 0);
    const int64_t nb = n_per_row / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[32];
    uint8_t Ls[QK_K/32];
    uint8_t Lm[QK_K/32];
    float   mins[QK_K/32];
    float   scales[QK_K/32];
    float   sw[QK_K/32];
    float   weights[32];
    struct k_sort k_sort;
    uint8_t k_buf[K_SORT_BUF_SIZE_LINEAR(QK_K/32, 0, 63)];

    k_sort_init_linear(&k_sort, QK_K/32, 0, 63, k_buf);

    for (int i = 0; i < nb; i++) {

        float sum_x2 = 0;
        for (int l = 0; l < QK_K; ++l) sum_x2 += x[l] * x[l];
        float sigma2 = 2*sum_x2/QK_K;
        float av_x = sqrtf(sigma2);

        for (int j = 0; j < QK_K/32; ++j) {
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*i + 32*j;
                for (int l = 0; l < 32; ++l) weights[l] = qw[l] * sqrtf(sigma2 + x[32*j + l]*x[32*j + l]);
            } else {
                for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
            }
            float sumw = 0;
            for (int l = 0; l < 32; ++l) sumw += weights[l];
            sw[j] = sumw;

            scales[j] = make_qkx3_quants(32, 31, x + 32*j, weights, L + 32*j, &mins[j], Laux, -0.9f, 0.05f, 36, false);
        }

        float d_block = make_qkxs_quants(QK_K/32, scales, sw, (int8_t *) Ls, (int8_t *) Laux, &k_sort, false, true);
        float m_block = make_qkxs_quants(QK_K/32, mins,   sw, (int8_t *) Lm, (int8_t *) Laux, &k_sort, false, true);
        // float d_block = make_qp_quants(QK_K/32, 63, scales, Ls, sw);
        // float m_block = make_qp_quants(QK_K/32, 63, mins,   Lm, sw);

        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t ls = Ls[j];
            uint8_t lm = Lm[j];
            ls = MIN(63, ls);
            lm = MIN(63, lm);
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } else {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d = GGML_FP32_TO_FP16(d_block);
        y[i].dmin = GGML_FP32_TO_FP16(m_block);

        // TODO: avoid requant
        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            if (!d) continue;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32*j + ii] + dm)/d);
                l = MAX(0, MIN(31, l));
                L[32*j + ii] = l;
            }
        }

        uint8_t * GGML_RESTRICT qh = y[i].qh;
        uint8_t * GGML_RESTRICT ql = y[i].qs;
        memset(qh, 0, QK_K/8);

        uint8_t m1 = 1, m2 = 2;
        for (int n = 0; n < QK_K; n += 64) {
            for (int j = 0; j < 32; ++j) {
                int l1 = L[n + j];
                if (l1 > 15) {
                    l1 -= 16; qh[j] |= m1;
                }
                int l2 = L[n + j + 32];
                if (l2 > 15) {
                    l2 -= 16; qh[j] |= m2;
                }
                ql[j] = l1 | (l2 << 4);
            }
            m1 <<= 2; m2 <<= 2;
            ql += 32;
        }

        x += QK_K;

    }
}

size_t quantize_q5_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    size_t row_size = ggml_row_size(GGML_TYPE_Q5_K, n_per_row);
    if (!quant_weights) {
        quantize_row_q5_K_ref(src, dst, (int64_t)nrow*n_per_row);
    }
    else {
        char * qrow = (char *)dst;
        for (int64_t row = 0; row < nrow; ++row) {
            quantize_row_q5_K_impl(src, (block_q5_K*)qrow, n_per_row, quant_weights);
            src += n_per_row;
            qrow += row_size;
        }
    }
    return nrow * row_size;
}

// ====================== 6-bit (de)-quantization

void quantize_row_q6_K_ref(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    int8_t L[QK_K];
    int8_t Laux[16];
    float  scales[QK_K/16];
    float  weights[16];

    struct k_sort k_sort;
    uint8_t k_buf[K_SORT_BUF_SIZE_LINEAR(16, -32, 31)];

    k_sort_init_linear(&k_sort, 16, -32, 31, k_buf);

    for (int i = 0; i < nb; i++) {

        float max_scale = 0;
        float max_abs_scale = 0;

        for (int ib = 0; ib < QK_K/16; ++ib) {
            for (int j = 0; j < 16; ++j) {
                weights[j] = x[16*ib + j] * x[16*ib + j];
            }

            const float scale = make_qkxs_quants(16, x + 16*ib, weights, L + 16*ib, Laux, &k_sort, true, true);
            // const float scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, NULL);
            scales[ib] = scale;

            const float abs_scale = fabsf(scale);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }

        }

        if (max_abs_scale < GROUP_MAX_EPS) {
            memset(&y[i], 0, sizeof(block_q6_K));
            y[i].d = GGML_FP32_TO_FP16(0.f);
            x += QK_K;
            continue;
        }

        float iscale = -128.f/max_scale;
        y[i].d = GGML_FP32_TO_FP16(1/iscale);
        for (int ib = 0; ib < QK_K/16; ++ib) {
            y[i].scales[ib] = MIN(127, nearest_int(iscale*scales[ib]));
        }

        // TODO: avoid requant
        for (int j = 0; j < QK_K/16; ++j) {
            float d = GGML_FP16_TO_FP32(y[i].d) * y[i].scales[j];
            if (!d) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16*j + ii]/d);
                l = MAX(-32, MIN(31, l));
                L[16*j + ii] = l + 32;
            }
        }

        uint8_t * GGML_RESTRICT ql = y[i].ql;
        uint8_t * GGML_RESTRICT qh = y[i].qh;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t q1 = L[j + l +  0] & 0xF;
                const uint8_t q2 = L[j + l + 32] & 0xF;
                const uint8_t q3 = L[j + l + 64] & 0xF;
                const uint8_t q4 = L[j + l + 96] & 0xF;
                ql[l+ 0] = q1 | (q3 << 4);
                ql[l+32] = q2 | (q4 << 4);
                qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
            }
            ql += 64;
            qh += 32;
        }

        x += QK_K;
    }
}

void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * GGML_RESTRICT ql = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const int8_t  * GGML_RESTRICT sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

static void quantize_row_q6_K_impl(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t n_per_row, const float * quant_weights) {
    assert(n_per_row % QK_K == 0);
    const int64_t nb = n_per_row / QK_K;

    int8_t L[QK_K];
    int8_t Laux[16];
    float  scales[QK_K/16];
    float  weights[16];

    struct k_sort k_sort;
    uint8_t k_buf[K_SORT_BUF_SIZE_LINEAR(16, -32, 31)];

    k_sort_init_linear(&k_sort, 16, -32, 31, k_buf);

    for (int i = 0; i < nb; i++) {

        //float sum_x2 = 0;
        //for (int j = 0; j < QK_K; ++j) sum_x2 += x[j]*x[j];
        //float sigma2 = sum_x2/QK_K;

        float max_scale = 0;
        float max_abs_scale = 0;

        for (int ib = 0; ib < QK_K/16; ++ib) {

            float scale;
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*i + 16*ib;
                //for (int j = 0; j < 16; ++j) weights[j] = qw[j] * sqrtf(sigma2 + x[16*ib + j]*x[16*ib + j]);
                //scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, weights);
                scale = make_qkxs_quants(16, x + 16*ib, qw, L + 16*ib, Laux, &k_sort, true, true);
            } else {
                for (int j = 0; j < 16; ++j) {
                    weights[j] = x[16*ib + j] * x[16*ib + j];
                }

                // scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, NULL);
                scale = make_qkxs_quants(16, x + 16*ib, weights, L + 16*ib, Laux, &k_sort, true, true);
            }
            scales[ib] = scale;

            const float abs_scale = fabsf(scale);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }

        }

        if (max_abs_scale < GROUP_MAX_EPS) {
            memset(&y[i], 0, sizeof(block_q6_K));
            y[i].d = GGML_FP32_TO_FP16(0.f);
            x += QK_K;
            continue;
        }

        float iscale = -128.f/max_scale;
        y[i].d = GGML_FP32_TO_FP16(1/iscale);
        for (int ib = 0; ib < QK_K/16; ++ib) {
            y[i].scales[ib] = MIN(127, nearest_int(iscale*scales[ib]));
        }

        // TODO: avoid requant
        for (int j = 0; j < QK_K/16; ++j) {
            float d = GGML_FP16_TO_FP32(y[i].d) * y[i].scales[j];
            if (!d) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16*j + ii]/d);
                l = MAX(-32, MIN(31, l));
                L[16*j + ii] = l + 32;
            }
        }

        uint8_t * GGML_RESTRICT ql = y[i].ql;
        uint8_t * GGML_RESTRICT qh = y[i].qh;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t q1 = L[j + l +  0] & 0xF;
                const uint8_t q2 = L[j + l + 32] & 0xF;
                const uint8_t q3 = L[j + l + 64] & 0xF;
                const uint8_t q4 = L[j + l + 96] & 0xF;
                ql[l+ 0] = q1 | (q3 << 4);
                ql[l+32] = q2 | (q4 << 4);
                qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
            }
            ql += 64;
            qh += 32;
        }

        x += QK_K;

    }
}

size_t quantize_q6_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    size_t row_size = ggml_row_size(GGML_TYPE_Q6_K, n_per_row);
    if (!quant_weights) {
        quantize_row_q6_K_ref(src, dst, (int64_t)nrow*n_per_row);
    }
    else {
        char * qrow = (char *)dst;
        for (int64_t row = 0; row < nrow; ++row) {
            quantize_row_q6_K_impl(src, (block_q6_K*)qrow, n_per_row, quant_weights);
            src += n_per_row;
            qrow += row_size;
        }
    }
    return nrow * row_size;
}

static void quantize_row_q4_0_impl(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t n_per_row, const float * quant_weights) {
    static_assert(QK4_0 == 32, "QK4_0 must be 32");

    if (!quant_weights) {
        quantize_row_q4_0_ref(x, y, n_per_row);
        return;
    }

    float weight[QK4_0];
    int8_t L[QK4_0];
    int8_t Laux[QK4_0];
    struct k_sort k_sort;
    uint8_t k_buf[K_SORT_BUF_SIZE_LINEAR(QK4_0, -8, 7)];

    k_sort_init_linear(&k_sort, QK4_0, -8, 7, k_buf);

    float sum_x2 = 0;
    for (int j = 0; j < n_per_row; ++j) sum_x2 += x[j]*x[j];
    float sigma2 = sum_x2/n_per_row;

    const int64_t nb = n_per_row/QK4_0;
    for (int ib = 0; ib < nb; ++ib) {
        const float * xb = x + QK4_0 * ib;
        const float * qw = quant_weights + QK4_0 * ib;
        for (int j = 0; j < QK4_0; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        float d = make_qkxs_quants(QK4_0, xb, weight, L, Laux, &k_sort, true, true);
        y[ib].d = GGML_FP32_TO_FP16(d);
        for (int j = 0; j < 16; ++j) {
            y[ib].qs[j] = L[j] | (L[j+16] << 4);
        }
    }
}

size_t quantize_q4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    if (!quant_weights) {
        quantize_row_q4_0_ref(src, dst, (int64_t)nrow*n_per_row);
        return nrow * ggml_row_size(GGML_TYPE_Q4_0, n_per_row);
    }
    size_t row_size = ggml_row_size(GGML_TYPE_Q4_0, n_per_row);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_q4_0_impl(src, (block_q4_0*)qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += row_size;
    }
    return nrow * row_size;
}

static void quantize_row_q4_1_impl(const float * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t n_per_row, const float * quant_weights) {
    static_assert(QK4_1 == 32, "QK4_1 must be 32");

    if (!quant_weights) {
        quantize_row_q4_1_ref(x, y, n_per_row);
        return;
    }

    float weight[QK4_1];
    uint8_t L[QK4_1], Laux[QK4_1];

    float sum_x2 = 0;
    for (int j = 0; j < n_per_row; ++j) sum_x2 += x[j]*x[j];
    float sigma2 = sum_x2/n_per_row;

    const int64_t nb = n_per_row/QK4_1;
    for (int ib = 0; ib < nb; ++ib) {
        const float * xb = x + QK4_1 * ib;
        const float * qw = quant_weights + QK4_1 * ib;
        for (int j = 0; j < QK4_1; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        float min;
        float d = make_qkx3_quants(QK4_1, 15, xb, weight, L, &min, Laux, -0.9f, 0.05f, 36, false);
        y[ib].d = GGML_FP32_TO_FP16(d);
        y[ib].m = GGML_FP32_TO_FP16(-min);
        for (int j = 0; j < 16; ++j) {
            y[ib].qs[j] = L[j] | (L[j+16] << 4);
        }
    }
}

size_t quantize_q4_1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    if (!quant_weights) {
        quantize_row_q4_1_ref(src, dst, (int64_t)nrow*n_per_row);
        return nrow * ggml_row_size(GGML_TYPE_Q4_1, n_per_row);
    }
    size_t row_size = ggml_row_size(GGML_TYPE_Q4_1, n_per_row);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_q4_1_impl(src, (block_q4_1*)qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += row_size;
    }
    return nrow * row_size;
}

static void quantize_row_q5_0_impl(const float * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t n_per_row, const float * quant_weights) {
    static_assert(QK5_0 == 32, "QK5_0 must be 32");

    if (!quant_weights) {
        quantize_row_q5_0_ref(x, y, n_per_row);
        return;
    }

    float weight[QK5_0];
    int8_t L[QK5_0];
    int8_t Laux[QK5_0];
    struct k_sort k_sort;
    uint8_t k_buf[K_SORT_BUF_SIZE_LINEAR(QK5_0, -16, 15)];

    k_sort_init_linear(&k_sort, QK5_0, -16, 15, k_buf);

    float sum_x2 = 0;
    for (int j = 0; j < n_per_row; ++j) sum_x2 += x[j]*x[j];
    float sigma2 = sum_x2/n_per_row;

    const int64_t nb = n_per_row/QK5_0;
    for (int ib = 0; ib < nb; ++ib) {
        const float * xb = x + QK5_0 * ib;
        const float * qw = quant_weights + QK5_0 * ib;
        for (int j = 0; j < QK5_0; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        float d = make_qkxs_quants(QK5_0, xb, weight, L, Laux, &k_sort, true, true);
        y[ib].d = GGML_FP32_TO_FP16(d);

        uint32_t qh = 0;

        for (int j = 0; j < 16; ++j) {
            const uint8_t xi0 = L[j];
            const uint8_t xi1 = L[j+16];
            y[ib].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            // get the 5-th bit and store it in qh at the right position
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
        }

        memcpy(&y[ib].qh, &qh, sizeof(qh));
    }
}

size_t quantize_q5_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    if (!quant_weights) {
        quantize_row_q5_0_ref(src, dst, (int64_t)nrow*n_per_row);
        return nrow * ggml_row_size(GGML_TYPE_Q5_0, n_per_row);
    }
    size_t row_size = ggml_row_size(GGML_TYPE_Q5_0, n_per_row);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_q5_0_impl(src, (block_q5_0*)qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += row_size;
    }
    return nrow * row_size;
}

static void quantize_row_q5_1_impl(const float * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t n_per_row, const float * quant_weights) {
    static_assert(QK5_1 == 32, "QK5_1 must be 32");

    if (!quant_weights) {
        quantize_row_q5_1_ref(x, y, n_per_row);
        return;
    }

    float weight[QK5_1];
    uint8_t L[QK5_1], Laux[QK5_1];

    float sum_x2 = 0;
    for (int j = 0; j < n_per_row; ++j) sum_x2 += x[j]*x[j];
    float sigma2 = sum_x2/n_per_row;

    const int64_t nb = n_per_row/QK5_1;
    for (int ib = 0; ib < nb; ++ib) {
        const float * xb = x + QK5_1 * ib;
        const float * qw = quant_weights + QK5_1 * ib;
        for (int j = 0; j < QK5_1; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        float min;
        float d = make_qkx3_quants(QK5_1, 31, xb, weight, L, &min, Laux, -0.9f, 0.05f, 36, false);
        y[ib].d = GGML_FP32_TO_FP16(d);
        y[ib].m = GGML_FP32_TO_FP16(-min);

        uint32_t qh = 0;
        for (int j = 0; j < 16; ++j) {
            const uint8_t xi0 = L[j];
            const uint8_t xi1 = L[j+16];
            y[ib].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            // get the 5-th bit and store it in qh at the right position
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
        }
        memcpy(&y[ib].qh, &qh, sizeof(qh));
    }
}

size_t quantize_q5_1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    if (!quant_weights) {
        quantize_row_q5_1_ref(src, dst, (int64_t)nrow*n_per_row);
        return nrow * ggml_row_size(GGML_TYPE_Q5_1, n_per_row);
    }
    size_t row_size = ggml_row_size(GGML_TYPE_Q5_1, n_per_row);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_q5_1_impl(src, (block_q5_1*)qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += row_size;
    }
    return nrow * row_size;
}

size_t quantize_q8_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void)quant_weights; // not used
    const size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);
    quantize_row_q8_0_ref(src, dst, (int64_t)nrow*n_per_row);
    return nrow * row_size;
}

size_t quantize_mxfp4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    GGML_UNUSED(quant_weights);
    quantize_row_mxfp4_ref(src, dst, (int64_t)nrow*n_per_row);
    return nrow * ggml_row_size(GGML_TYPE_MXFP4, n_per_row);
}

// ====================== Ternary (de)-quantization (BitNet b1.58 and TriLMs)

void quantize_row_tq1_0_ref(const float * GGML_RESTRICT x, block_tq1_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK_K; j++) {
            const float v = x[j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        // 5 elements per byte, along 32 bytes
        for (size_t j = 0; j < sizeof(y->qs) - sizeof(y->qs) % 32; j += 32) {
            for (size_t m = 0; m < 32; ++m) {
                uint8_t q = 0;
                for (size_t n = 0; n < 5; ++n) {
                    int xi = lroundf(x[m + n*32] * id) + 1; // -1, 0, 1 -> 0, 1, 2
                    q *= 3;
                    q += xi;
                }
                // ceiling division (243 == pow(3, 5))
                q = ((uint16_t)q * 256 + (243 - 1)) / 243;
                y[i].qs[j + m] = q;
            }
            x += 5*32;
        }
        // along 16 bytes
        for (size_t j = sizeof(y->qs) - sizeof(y->qs) % 32; j < sizeof(y->qs); j += 16) {
            for (size_t m = 0; m < 16; ++m) {
                uint8_t q = 0;
                for (size_t n = 0; n < 5; ++n) {
                    int xi = lroundf(x[m + n*16] * id) + 1; // -1, 0, 1 -> 0, 1, 2
                    q *= 3;
                    q += xi;
                }
                // ceiling division (243 == pow(3, 5))
                q = ((uint16_t)q * 256 + (243 - 1)) / 243;
                y[i].qs[j + m] = q;
            }
            x += 5*16;
        }
        // 4 elements per byte
        for (size_t j = 0; j < sizeof(y->qh); ++j) {
            uint8_t q = 0;
            for (size_t m = 0; m < 4; ++m) {
                // -1, 0, 1 -> 0, 1, 2
                int xi = lroundf(x[j + m*sizeof(y->qh)] * id) + 1;
                q *= 3;
                q += xi;
            }
            // shift the first value to the most significant trit
            q *= 3;
            // ceiling division (243 == pow(3, 5))
            q = ((uint16_t)q * 256 + (243 - 1)) / 243;
            y[i].qh[j] = q;
        }
        x += 4*sizeof(y->qh);
    }
}

static void quantize_row_tq1_0_impl(const float * GGML_RESTRICT x, block_tq1_0 * GGML_RESTRICT y, int64_t n_per_row, const float * quant_weights) {
    if (!quant_weights) {
        quantize_row_tq1_0_ref(x, y, n_per_row);
        return;
    }

    float weight[QK_K];
    int8_t L[QK_K];
    int8_t Laux[QK_K];
    struct k_sort k_sort;
    uint8_t k_buf[K_SORT_BUF_SIZE_LINEAR(QK_K, -1, 1)];

    k_sort_init_linear(&k_sort, QK_K, -1, 1, k_buf);

    float sum_x2 = 0;
    for (int j = 0; j < n_per_row; ++j) { sum_x2 += x[j]*x[j]; }
    float sigma2 = sum_x2/n_per_row;

    const int64_t nb = n_per_row/QK_K;
    for (int ib = 0; ib < nb; ++ib) {
        const float * xb = x + QK_K * ib;
        const float * qw = quant_weights + QK_K * ib;
        const int8_t * Lptr = L;
        for (int j = 0; j < QK_K; ++j) { weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]); }
        float d = make_qkxs_quants(QK_K, xb, weight, L, Laux, &k_sort, false, false);
        y[ib].d = GGML_FP32_TO_FP16(d);

        // 5 elements per byte, along 32 bytes
        for (size_t j = 0; j < sizeof(y->qs) - sizeof(y->qs) % 32; j += 32) {
            for (size_t m = 0; m < 32; ++m) {
                uint8_t q = 0;
                for (size_t n = 0; n < 5; ++n) {
                    q *= 3;
                    q += Lptr[m + n*32];
                }
                // ceiling division (243 == pow(3, 5))
                q = ((uint16_t)q * 256 + (243 - 1)) / 243;
                y[ib].qs[j + m] = q;
            }
            Lptr += 5*32;
        }
        // along 16 bytes
        for (size_t j = sizeof(y->qs) - sizeof(y->qs) % 32; j < sizeof(y->qs); j += 16) {
            for (size_t m = 0; m < 16; ++m) {
                uint8_t q = 0;
                for (size_t n = 0; n < 5; ++n) {
                    q *= 3;
                    q += Lptr[m + n*16];
                }
                // ceiling division (243 == pow(3, 5))
                q = ((uint16_t)q * 256 + (243 - 1)) / 243;
                y[ib].qs[j + m] = q;
            }
            Lptr += 5*16;
        }
        // 4 elements per byte
        for (size_t j = 0; j < sizeof(y->qh); ++j) {
            uint8_t q = 0;
            for (size_t m = 0; m < 4; ++m) {
                q *= 3;
                q += Lptr[j + m*sizeof(y->qh)];
            }
            // shift the first value to the most significant trit
            q *= 3;
            // ceiling division (243 == pow(3, 5))
            q = ((uint16_t)q * 256 + (243 - 1)) / 243;
            y[ib].qh[j] = q;
        }
    }
}

void quantize_row_tq2_0_ref(const float * GGML_RESTRICT x, block_tq2_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK_K; j++) {
            const float v = x[j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (size_t j = 0; j < sizeof(y->qs); j += 32) {
            for (size_t m = 0; m < 32; ++m) {
                uint8_t q = 0;
                for (size_t n = 0; n < 4; ++n) {
                    // -1, 0, 1 -> 0, 1, 2
                    int xi = lroundf(x[m + n*32] * id) + 1;
                    q += (xi & 3) << (2*n);
                }
                y[i].qs[j + m] = q;
            }
            x += 4*32;
        }
    }
}


static void quantize_row_tq2_0_impl(const float * GGML_RESTRICT x, block_tq2_0 * GGML_RESTRICT y, int64_t n_per_row, const float * quant_weights) {
    if (!quant_weights) {
        quantize_row_tq2_0_ref(x, y, n_per_row);
        return;
    }

    float weight[QK_K];
    int8_t L[QK_K];
    int8_t Laux[QK_K];
    struct k_sort k_sort;
    uint8_t k_buf[K_SORT_BUF_SIZE_LINEAR(QK_K, -1, 2)];

    k_sort_init_linear(&k_sort, QK_K, -1, 2, k_buf);

    float sum_x2 = 0;
    for (int j = 0; j < n_per_row; ++j) { sum_x2 += x[j]*x[j]; }
    float sigma2 = sum_x2/n_per_row;

    const int64_t nb = n_per_row/QK_K;
    for (int ib = 0; ib < nb; ++ib) {
        const float * xb = x + QK_K * ib;
        const float * qw = quant_weights + QK_K * ib;
        for (int j = 0; j < QK_K; ++j) { weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]); }
        float d = make_qkxs_quants(QK_K, xb, weight, L, Laux, &k_sort, true, false);
        y[ib].d = GGML_FP32_TO_FP16(d);

        for (size_t j = 0; j < sizeof(y->qs); j += 32) {
            for (size_t m = 0; m < 32; ++m) {
                uint8_t q = 0;
                for (size_t n = 0; n < 4; ++n) {
                    q += (L[4*j + m + n*32] & 3) << (2*n);
                }
                y[ib].qs[j + m] = q;
            }
        }
    }
}

size_t quantize_tq1_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    if (!quant_weights) {
        quantize_row_tq1_0_ref(src, dst, (int64_t)nrow*n_per_row);
        return nrow * ggml_row_size(GGML_TYPE_TQ1_0, n_per_row);
    }
    size_t row_size = ggml_row_size(GGML_TYPE_TQ1_0, n_per_row);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_tq1_0_impl(src, (block_tq1_0*)qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += row_size;
    }
    return nrow * row_size;
}

size_t quantize_tq2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    if (!quant_weights) {
        quantize_row_tq2_0_ref(src, dst, (int64_t)nrow*n_per_row);
        return nrow * ggml_row_size(GGML_TYPE_TQ2_0, n_per_row);
    }
    size_t row_size = ggml_row_size(GGML_TYPE_TQ2_0, n_per_row);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_tq2_0_impl(src, (block_tq2_0*)qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += row_size;
    }
    return nrow * row_size;
}

void dequantize_row_tq1_0(const block_tq1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};

    for (int64_t i = 0; i < nb; ++i) {

        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (size_t j = 0; j < sizeof(x->qs) - sizeof(x->qs) % 32; j += 32) {
            for (size_t n = 0; n < 5; ++n) {
                for (size_t m = 0; m < 32; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[n];
                    int16_t xi = ((uint16_t) q * 3) >> 8;
                    *y++ = (float) (xi - 1) * d;
                }
            }
        }
        for (size_t j = sizeof(x->qs) - sizeof(x->qs) % 32; j < sizeof(x->qs); j += 16) {
            for (size_t n = 0; n < 5; ++n) {
                for (size_t m = 0; m < 16; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[n];
                    int16_t xi = ((uint16_t) q * 3) >> 8;
                    *y++ = (float) (xi - 1) * d;
                }
            }
        }

        for (size_t n = 0; n < 4; ++n) {
            for (size_t j = 0; j < sizeof(x->qh); ++j) {
                uint8_t q = x[i].qh[j] * pow3[n];
                int16_t xi = ((uint16_t) q * 3) >> 8;
                *y++ = (float) (xi - 1) * d;
            }
        }
    }
}

void dequantize_row_tq2_0(const block_tq2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; ++i) {

        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (size_t j = 0; j < sizeof(x->qs); j += 32) {
            for (size_t l = 0; l < 4; ++l) {
                for (size_t m = 0; m < 32; ++m) {
                    int8_t q = (x[i].qs[j + m] >> (l*2)) & 3;
                    *y++ = (float) (q - 1) * d;
                }
            }
        }
    }
}

// ====================== "True" 2-bit (de)-quantization

void dequantize_row_iq2_xxs(const block_iq2_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(aux32, x[i].qs + 4*ib32, 2*sizeof(uint32_t));
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}

// ====================== 2.3125 bpw (de)-quantization

void dequantize_row_iq2_xs(const block_iq2_xs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    float db[2];

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >>  4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (x[i].qs[4*ib32 + l] & 511));
                const uint8_t  signs = ksigns_iq2xs[x[i].qs[4*ib32 + l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db[l/2] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}

// ====================== 2.5625 bpw (de)-quantization

void dequantize_row_iq2_s(const block_iq2_s * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    float db[2];

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * signs = qs + QK_K/8;

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >>  4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const float dl = db[l/2];
                const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 4;
            signs += 4;
        }
    }
}

// ====================== 3.0625 bpw (de)-quantization

void dequantize_row_iq3_xxs(const block_iq3_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    uint32_t aux32;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * scales_and_signs = qs + QK_K/4;

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(&aux32, scales_and_signs + 4*ib32, sizeof(uint32_t));
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t  signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid + qs[2*l+0]);
                const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid + qs[2*l+1]);
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    y[j+4] = db * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
        }
    }
}

// ====================== 3.3125 bpw (de)-quantization

void dequantize_row_iq3_s(const block_iq3_s * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * signs = x[i].signs;

        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const float db1 = d * (1 + 2*(x[i].scales[ib32/2] & 0xf));
            const float db2 = d * (1 + 2*(x[i].scales[ib32/2] >>  4));
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[0] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[0] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db1 * grid1[j] * (signs[l] & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    y[j+4] = db1 * grid2[j] * (signs[l] & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[1] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[1] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db2 * grid1[j] * (signs[l] & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    y[j+4] = db2 * grid2[j] * (signs[l] & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
}

// ====================== 1.5625 bpw (de)-quantization

void dequantize_row_iq1_s(const block_iq1_s * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t  * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const float dl = d * (2*((qh[ib] >> 12) & 7) + 1);
            const float delta = qh[ib] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
            for (int l = 0; l < 4; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + (qs[l] | (((qh[ib] >> 3*l) & 7) << 8)));
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl * (grid[j] + delta);
                }
                y += 8;
            }
            qs += 4;
        }
    }
}

void dequantize_row_iq1_m(const block_iq1_m * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    float delta[4];
    uint16_t idx[4];

    iq1m_scale_t scale;

    for (int i = 0; i < nb; i++) {

        const uint16_t * sc = (const uint16_t *)x[i].scales;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        const float d = GGML_FP16_TO_FP32(scale.f16);

        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const float dl1 = d * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1);
            const float dl2 = d * (2*((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1);

            idx[0] = qs[0] | ((qh[0] << 8) & 0x700);
            idx[1] = qs[1] | ((qh[0] << 4) & 0x700);
            idx[2] = qs[2] | ((qh[1] << 8) & 0x700);
            idx[3] = qs[3] | ((qh[1] << 4) & 0x700);
            delta[0] = qh[0] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[1] = qh[0] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[2] = qh[1] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[3] = qh[1] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
            for (int l = 0; l < 2; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + idx[l]);
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl1 * (grid[j] + delta[l]);
                }
                y += 8;
            }
            for (int l = 2; l < 4; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + idx[l]);
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl2 * (grid[j] + delta[l]);
                }
                y += 8;
            }
            qs += 4;
            qh += 2;
        }
    }
}

void dequantize_row_iq4_nl(const block_iq4_nl * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK4_NL == 0);
    const int64_t nb = k / QK4_NL;

    for (int i = 0; i < nb; i++) {

        const uint8_t * qs = x[i].qs;

        const float d = GGML_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < QK4_NL/2; ++j) {
            y[j+       0] = d * kvalues_iq4nl[qs[j] & 0xf];
            y[j+QK4_NL/2] = d * kvalues_iq4nl[qs[j] >>  4];
        }
        y  += QK4_NL;
        qs += QK4_NL/2;
    }
}

void dequantize_row_iq4_xs(const block_iq4_xs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const uint8_t * qs = x[i].qs;

        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const int ls = ((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl * kvalues_iq4nl[qs[j] & 0xf];
                y[j+16] = dl * kvalues_iq4nl[qs[j] >>  4];
            }
            y  += 32;
            qs += 16;
        }
    }
}

//===================================== Q8_K ==============================================

void quantize_row_q8_K_ref(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        float max = 0;
        float amax = 0;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax; max = x[j];
            }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, QK_K);
            x += QK_K;
            continue;
        }
        //const float iscale = -128.f/max;
        // We need this change for IQ2_XXS, else the AVX implementation becomes very awkward
        const float iscale = -127.f/max;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int(iscale*x[j]);
            y[i].qs[j] = MIN(127, v);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j*16 + ii];
            }
            y[i].bsums[j] = sum;
        }
        y[i].d = 1/iscale;
        x += QK_K;
    }
}

void dequantize_row_q8_K(const block_q8_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < QK_K; ++j) {
            *y++ = x[i].d * x[i].qs[j];
        }
    }
}

// ================================ IQ2 quantization =============================================

typedef struct {
    uint64_t * grid;
    int      * map;
    uint16_t * neighbours;
} iq2_entry_t;

static iq2_entry_t iq2_data[4] = {
    { NULL, NULL, NULL }, // IQ2_XXS
    { NULL, NULL, NULL }, // IQ2_XS
    { NULL, NULL, NULL }, // IQ1_S IQ1_M
    { NULL, NULL, NULL }, // IQ2_S
};

static inline int iq2_data_index(enum ggml_type type) {
    GGML_ASSERT(type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M || type == GGML_TYPE_IQ2_S);
    return type == GGML_TYPE_IQ2_XXS ? 0 :
           type == GGML_TYPE_IQ2_XS  ? 1 :
           type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M ? 2 : 3;
}

static inline int iq2_grid_size(enum ggml_type type) {
    GGML_ASSERT(type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M || type == GGML_TYPE_IQ2_S);
    return type == GGML_TYPE_IQ2_XXS ? 256 :
           type == GGML_TYPE_IQ2_XS  ? 512 :
           type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M ? NGRID_IQ1S : 1024;
}

static int iq2_compare_func(const void * left, const void * right) {
    const int * l = (const int *)left;
    const int * r = (const int *)right;
    return l[0] < r[0] ? -1 : l[0] > r[0] ? 1 : l[1] < r[1] ? -1 : l[1] > r[1] ? 1 : 0;
}

void iq2xs_init_impl(enum ggml_type type) {
    const int gindex = iq2_data_index(type);
    const int grid_size = iq2_grid_size(type);
    if (iq2_data[gindex].map) {
        return;
    }
    const uint16_t pow3[9] = { 1, 3, 9, 27, 81, 243, 729, 2187, 6561 };
    // TODO: maybe repack the constants to fit in pow3[8]
    static const uint16_t kgrid_2bit_256[256] = {
            0,     2,     5,     8,    10,    17,    20,    32,    34,    40,    42,    65,    68,    80,    88,    97,
          100,   128,   130,   138,   162,   257,   260,   272,   277,   320,   388,   408,   512,   514,   546,   642,
         1025,  1028,  1040,  1057,  1060,  1088,  1090,  1096,  1120,  1153,  1156,  1168,  1188,  1280,  1282,  1288,
         1312,  1350,  1385,  1408,  1425,  1545,  1552,  1600,  1668,  1700,  2048,  2053,  2056,  2068,  2088,  2113,
         2116,  2128,  2130,  2184,  2308,  2368,  2562,  2580,  4097,  4100,  4112,  4129,  4160,  4192,  4228,  4240,
         4245,  4352,  4360,  4384,  4432,  4442,  4480,  4644,  4677,  5120,  5128,  5152,  5157,  5193,  5248,  5400,
         5474,  5632,  5654,  6145,  6148,  6160,  6208,  6273,  6400,  6405,  6560,  6737,  8192,  8194,  8202,  8260,
         8289,  8320,  8322,  8489,  8520,  8704,  8706,  9217,  9220,  9232,  9280,  9302,  9472,  9537,  9572,  9872,
        10248, 10272, 10388, 10820, 16385, 16388, 16400, 16408, 16417, 16420, 16448, 16456, 16470, 16480, 16513, 16516,
        16528, 16640, 16672, 16737, 16768, 16773, 16897, 16912, 16968, 16982, 17000, 17408, 17416, 17440, 17536, 17561,
        17682, 17700, 17920, 18433, 18436, 18448, 18496, 18501, 18688, 18776, 18785, 18818, 19013, 19088, 20480, 20488,
        20497, 20505, 20512, 20608, 20616, 20740, 20802, 20900, 21137, 21648, 21650, 21770, 22017, 22100, 22528, 22545,
        22553, 22628, 22848, 23048, 24580, 24592, 24640, 24680, 24832, 24917, 25112, 25184, 25600, 25605, 25872, 25874,
        25988, 26690, 32768, 32770, 32778, 32833, 32898, 33028, 33048, 33088, 33297, 33793, 33796, 33808, 33813, 33856,
        33888, 34048, 34118, 34196, 34313, 34368, 34400, 34818, 35076, 35345, 36868, 36880, 36900, 36928, 37025, 37142,
        37248, 37445, 37888, 37922, 37956, 38225, 39041, 39200, 40962, 41040, 41093, 41225, 41472, 42008, 43088, 43268,
    };
    static const uint16_t kgrid_2bit_512[512] = {
            0,     2,     5,     8,    10,    17,    20,    22,    25,    32,    34,    37,    40,    65,    68,    70,
           73,    80,    82,    85,    88,    97,   100,   128,   130,   133,   136,   145,   148,   153,   160,   257,
          260,   262,   265,   272,   274,   277,   280,   282,   289,   292,   320,   322,   325,   328,   337,   340,
          352,   360,   385,   388,   400,   512,   514,   517,   520,   529,   532,   544,   577,   580,   592,   597,
          640,   650,  1025,  1028,  1030,  1033,  1040,  1042,  1045,  1048,  1057,  1060,  1088,  1090,  1093,  1096,
         1105,  1108,  1110,  1120,  1153,  1156,  1168,  1280,  1282,  1285,  1288,  1297,  1300,  1312,  1345,  1348,
         1360,  1377,  1408,  1537,  1540,  1552,  1574,  1600,  1602,  1668,  2048,  2050,  2053,  2056,  2058,  2065,
         2068,  2080,  2085,  2113,  2116,  2128,  2136,  2176,  2208,  2218,  2305,  2308,  2320,  2368,  2433,  2441,
         2560,  2592,  2600,  2710,  2720,  4097,  4100,  4102,  4105,  4112,  4114,  4117,  4120,  4129,  4132,  4160,
         4162,  4165,  4168,  4177,  4180,  4192,  4202,  4225,  4228,  4240,  4352,  4354,  4357,  4360,  4369,  4372,
         4384,  4417,  4420,  4432,  4480,  4500,  4502,  4609,  4612,  4614,  4624,  4672,  4704,  5120,  5122,  5125,
         5128,  5137,  5140,  5152,  5185,  5188,  5193,  5200,  5220,  5248,  5377,  5380,  5392,  5440,  5632,  5652,
         5705,  6145,  6148,  6160,  6162,  6208,  6228,  6278,  6400,  6405,  6502,  6737,  6825,  8192,  8194,  8197,
         8200,  8202,  8209,  8212,  8224,  8257,  8260,  8272,  8320,  8352,  8449,  8452,  8464,  8512,  8520,  8549,
         8704,  8738,  8832,  8872,  9217,  9220,  9232,  9257,  9280,  9472,  9537,  9554,  9625,  9729,  9754,  9894,
        10240, 10248, 10250, 10272, 10325, 10376, 10402, 10600, 10640, 10760, 10784, 10882, 10888, 10890, 16385, 16388,
        16390, 16393, 16400, 16402, 16405, 16408, 16417, 16420, 16448, 16450, 16453, 16456, 16458, 16465, 16468, 16480,
        16485, 16513, 16516, 16528, 16640, 16642, 16645, 16648, 16657, 16660, 16672, 16705, 16708, 16720, 16768, 16773,
        16802, 16897, 16900, 16912, 16914, 16937, 16960, 17408, 17410, 17413, 17416, 17425, 17428, 17433, 17440, 17473,
        17476, 17488, 17536, 17556, 17665, 17668, 17680, 17700, 17728, 17818, 17920, 17930, 17988, 18000, 18433, 18436,
        18448, 18496, 18501, 18516, 18530, 18688, 18705, 18756, 18768, 18793, 18948, 20480, 20482, 20485, 20488, 20497,
        20500, 20512, 20520, 20545, 20548, 20560, 20608, 20737, 20740, 20752, 20757, 20800, 20802, 20992, 21060, 21162,
        21505, 21508, 21520, 21537, 21568, 21600, 21633, 21665, 21760, 21768, 21888, 21896, 22049, 22120, 22177, 22528,
        22548, 22593, 22608, 22681, 22810, 22848, 22850, 23173, 24577, 24580, 24592, 24640, 24660, 24674, 24710, 24745,
        24832, 25124, 25162, 25234, 25600, 25622, 25872, 25920, 25925, 26020, 26625, 26730, 26917, 27142, 27220, 27234,
        32768, 32770, 32773, 32776, 32785, 32788, 32800, 32810, 32833, 32836, 32848, 32896, 32898, 32936, 32938, 33025,
        33028, 33030, 33040, 33088, 33105, 33113, 33280, 33312, 33408, 33410, 33440, 33448, 33793, 33796, 33808, 33810,
        33813, 33856, 33888, 33929, 34048, 34116, 34213, 34328, 34410, 34816, 34824, 34853, 34906, 34944, 34946, 34984,
        35078, 35362, 35456, 35464, 35478, 35496, 36865, 36868, 36880, 36928, 36950, 36996, 37120, 37154, 37220, 37462,
        37513, 37888, 37893, 37956, 37968, 37976, 38185, 38288, 38290, 38465, 38993, 39078, 39241, 39445, 39520, 40960,
        40962, 40968, 40970, 40992, 41002, 41120, 41297, 41305, 41382, 41472, 41474, 41480, 41514, 41600, 41632, 42048,
        42133, 42597, 42648, 43018, 43040, 43042, 43048, 43168, 43176, 43268, 43396, 43398, 43560, 43562, 43665, 43690,
    };
    static const uint16_t kgrid_1bit_2048[NGRID_IQ1S] = {
            0,     2,     5,     8,    10,    17,    21,    32,    34,    40,    42,    69,    81,    84,    86,   101,
          128,   130,   136,   138,   149,   160,   162,   168,   170,   260,   261,   273,   276,   278,   281,   282,
          293,   321,   326,   329,   338,   341,   346,   353,   356,   358,   360,   389,   401,   404,   406,   421,
          512,   514,   520,   522,   533,   544,   546,   552,   554,   581,   593,   601,   612,   617,   640,   642,
          648,   650,   657,   661,   665,   672,   674,   680,   682,  1041,  1044,  1046,  1061,  1089,  1097,  1109,
         1114,  1124,  1125,  1169,  1177,  1189,  1281,  1284,  1285,  1286,  1301,  1304,  1306,  1321,  1344,  1349,
         1354,  1360,  1361,  1364,  1365,  1366,  1369,  1376,  1378,  1381,  1384,  1386,  1409,  1425,  1429,  1432,
         1434,  1441,  1444,  1445,  1446,  1449,  1556,  1561,  1601,  1604,  1616,  1618,  1621,  1624,  1632,  1633,
         1638,  1641,  1669,  1681,  1684,  1689,  2048,  2050,  2056,  2058,  2069,  2080,  2082,  2088,  2090,  2117,
         2129,  2134,  2149,  2176,  2178,  2184,  2186,  2197,  2208,  2210,  2216,  2218,  2309,  2321,  2324,  2329,
         2340,  2341,  2369,  2384,  2385,  2389,  2401,  2404,  2409,  2449,  2452,  2454,  2457,  2469,  2560,  2562,
         2568,  2570,  2581,  2592,  2594,  2600,  2602,  2629,  2641,  2649,  2657,  2661,  2688,  2690,  2693,  2696,
         2698,  2709,  2720,  2722,  2728,  2730,  4112,  4113,  4116,  4121,  4132,  4133,  4161,  4164,  4176,  4181,
         4184,  4193,  4196,  4197,  4201,  4241,  4244,  4246,  4257,  4261,  4353,  4356,  4358,  4361,  4368,  4370,
         4373,  4376,  4385,  4388,  4393,  4421,  4426,  4432,  4433,  4434,  4436,  4437,  4438,  4441,  4448,  4453,
         4484,  4498,  4501,  4513,  4516,  4625,  4628,  4630,  4645,  4672,  4678,  4681,  4690,  4693,  4696,  4698,
         4708,  4710,  4741,  4753,  4756,  4758,  4773,  5121,  5126,  5129,  5140,  5141,  5144,  5145,  5153,  5158,
         5185,  5189,  5190,  5192,  5194,  5201,  5204,  5205,  5206,  5209,  5218,  5221,  5224,  5252,  5257,  5264,
         5268,  5269,  5272,  5273,  5274,  5281,  5284,  5285,  5289,  5378,  5381,  5386,  5393,  5396,  5397,  5398,
         5401,  5408,  5410,  5413,  5416,  5418,  5441,  5444,  5445,  5446,  5457,  5458,  5460,  5461,  5462,  5465,
         5466,  5473,  5476,  5477,  5478,  5481,  5504,  5506,  5508,  5509,  5512,  5514,  5520,  5521,  5524,  5525,
         5526,  5529,  5530,  5536,  5538,  5541,  5633,  5636,  5637,  5638,  5653,  5654,  5656,  5658,  5665,  5670,
         5696,  5698,  5700,  5701,  5704,  5706,  5713,  5717,  5718,  5720,  5721,  5729,  5732,  5733,  5736,  5737,
         5738,  5766,  5770,  5778,  5781,  5796,  5801,  6161,  6166,  6181,  6209,  6212,  6214,  6217,  6224,  6229,
         6232,  6234,  6240,  6241,  6244,  6246,  6249,  6277,  6289,  6292,  6309,  6416,  6418,  6421,  6426,  6433,
         6437,  6466,  6468,  6469,  6472,  6481,  6484,  6485,  6486,  6489,  6490,  6496,  6501,  6506,  6537,  6545,
         6546,  6549,  6552,  6561,  6566,  6569,  6665,  6678,  6692,  6694,  6724,  6726,  6729,  6736,  6738,  6741,
         6744,  6753,  6758,  6761,  6789,  6801,  6806,  6810,  8192,  8194,  8200,  8202,  8213,  8224,  8226,  8229,
         8232,  8234,  8261,  8273,  8281,  8289,  8293,  8320,  8322,  8328,  8330,  8341,  8352,  8354,  8357,  8360,
         8362,  8453,  8465,  8468,  8473,  8485,  8514,  8516,  8521,  8533,  8536,  8538,  8545,  8548,  8549,  8550,
         8581,  8592,  8598,  8601,  8613,  8705,  8712,  8714,  8721,  8725,  8736,  8738,  8744,  8746,  8773,  8785,
         8790,  8793,  8805,  8833,  8840,  8842,  8849,  8853,  8864,  8866,  8872,  8874,  9221,  9236,  9238,  9241,
         9253,  9284,  9285,  9286,  9289,  9298,  9301,  9304,  9306,  9318,  9349,  9361,  9364,  9369,  9377,  9381,
         9481,  9493,  9505,  9513,  9536,  9541,  9544,  9553,  9556,  9557,  9561,  9570,  9573,  9576,  9609,  9616,
         9620,  9621,  9624,  9626,  9633,  9636,  9638,  9641,  9733,  9744,  9746,  9753,  9765,  9793,  9801,  9813,
         9824,  9825,  9833,  9860,  9862,  9872,  9882, 10240, 10242, 10248, 10250, 10261, 10272, 10274, 10280, 10282,
        10309, 10321, 10324, 10341, 10368, 10370, 10376, 10378, 10400, 10402, 10408, 10410, 10505, 10513, 10516, 10521,
        10533, 10566, 10569, 10578, 10581, 10593, 10596, 10598, 10601, 10629, 10640, 10646, 10649, 10660, 10661, 10752,
        10754, 10760, 10762, 10784, 10786, 10792, 10794, 10821, 10833, 10838, 10841, 10853, 10880, 10882, 10888, 10890,
        10901, 10912, 10914, 10920, 10922, 16389, 16401, 16406, 16421, 16457, 16466, 16469, 16472, 16474, 16481, 16484,
        16486, 16532, 16537, 16545, 16550, 16640, 16641, 16644, 16646, 16649, 16658, 16661, 16662, 16664, 16666, 16673,
        16678, 16681, 16709, 16712, 16714, 16721, 16724, 16725, 16726, 16729, 16730, 16741, 16744, 16746, 16769, 16772,
        16774, 16784, 16786, 16789, 16800, 16801, 16802, 16901, 16913, 16916, 16918, 16933, 16961, 16978, 16981, 16986,
        16996, 17001, 17033, 17044, 17061, 17409, 17429, 17433, 17449, 17477, 17480, 17482, 17489, 17492, 17493, 17494,
        17505, 17506, 17509, 17512, 17514, 17537, 17542, 17545, 17552, 17554, 17557, 17568, 17569, 17577, 17665, 17666,
        17669, 17674, 17681, 17684, 17685, 17686, 17689, 17696, 17701, 17706, 17729, 17732, 17733, 17734, 17737, 17744,
        17745, 17748, 17749, 17750, 17752, 17753, 17761, 17764, 17765, 17766, 17769, 17794, 17796, 17797, 17800, 17809,
        17812, 17813, 17814, 17817, 17818, 17829, 17832, 17834, 17921, 17925, 17929, 17940, 17941, 17944, 17946, 17953,
        17956, 17961, 17984, 17986, 17989, 17992, 18000, 18001, 18002, 18005, 18006, 18009, 18018, 18021, 18024, 18049,
        18053, 18058, 18068, 18069, 18081, 18084, 18086, 18437, 18449, 18453, 18458, 18469, 18498, 18505, 18512, 18517,
        18520, 18529, 18532, 18534, 18537, 18565, 18577, 18580, 18582, 18585, 18597, 18689, 18693, 18694, 18698, 18704,
        18708, 18709, 18712, 18721, 18724, 18726, 18752, 18757, 18762, 18769, 18770, 18772, 18773, 18774, 18777, 18784,
        18786, 18789, 18790, 18794, 18822, 18825, 18834, 18837, 18838, 18840, 18849, 18852, 18854, 18857, 18966, 19012,
        19014, 19017, 19029, 19032, 19034, 19044, 19049, 19092, 19109, 20481, 20484, 20485, 20486, 20489, 20498, 20501,
        20506, 20513, 20516, 20521, 20544, 20549, 20552, 20561, 20564, 20565, 20566, 20569, 20581, 20584, 20614, 20617,
        20629, 20632, 20640, 20641, 20646, 20649, 20741, 20744, 20745, 20746, 20753, 20756, 20757, 20758, 20760, 20761,
        20768, 20773, 20774, 20776, 20778, 20801, 20804, 20805, 20806, 20809, 20816, 20817, 20818, 20820, 20821, 20822,
        20824, 20825, 20826, 20833, 20836, 20837, 20838, 20841, 20866, 20869, 20881, 20884, 20885, 20886, 20889, 20896,
        20901, 20906, 20993, 20998, 21010, 21013, 21018, 21025, 21028, 21058, 21061, 21066, 21073, 21076, 21077, 21078,
        21081, 21090, 21093, 21125, 21136, 21138, 21141, 21145, 21146, 21156, 21508, 21509, 21521, 21524, 21525, 21526,
        21528, 21529, 21537, 21541, 21544, 21546, 21569, 21572, 21573, 21574, 21577, 21578, 21584, 21585, 21588, 21589,
        21590, 21592, 21593, 21594, 21601, 21602, 21604, 21605, 21606, 21609, 21632, 21640, 21642, 21649, 21652, 21653,
        21654, 21657, 21665, 21668, 21669, 21674, 21761, 21762, 21764, 21765, 21766, 21769, 21776, 21777, 21778, 21780,
        21781, 21782, 21785, 21786, 21793, 21796, 21797, 21798, 21801, 21824, 21825, 21826, 21828, 21829, 21830, 21832,
        21833, 21840, 21841, 21842, 21844, 21845, 21846, 21848, 21849, 21850, 21856, 21857, 21860, 21861, 21862, 21864,
        21865, 21866, 21889, 21892, 21893, 21897, 21898, 21904, 21905, 21908, 21909, 21910, 21912, 21913, 21921, 21924,
        21925, 21926, 21929, 22016, 22017, 22018, 22020, 22022, 22024, 22025, 22033, 22036, 22037, 22040, 22041, 22048,
        22049, 22050, 22052, 22053, 22054, 22056, 22057, 22081, 22085, 22086, 22088, 22089, 22090, 22096, 22097, 22098,
        22100, 22101, 22102, 22104, 22105, 22106, 22113, 22116, 22117, 22121, 22146, 22149, 22150, 22152, 22153, 22154,
        22161, 22165, 22170, 22178, 22181, 22182, 22184, 22185, 22532, 22533, 22534, 22537, 22544, 22549, 22552, 22561,
        22570, 22597, 22600, 22602, 22609, 22612, 22613, 22614, 22616, 22617, 22624, 22626, 22628, 22629, 22658, 22665,
        22672, 22674, 22677, 22680, 22689, 22697, 22785, 22786, 22789, 22794, 22801, 22804, 22805, 22806, 22809, 22821,
        22849, 22852, 22853, 22854, 22857, 22864, 22865, 22866, 22868, 22869, 22870, 22872, 22873, 22874, 22881, 22884,
        22885, 22886, 22889, 22913, 22917, 22921, 22929, 22932, 22933, 22934, 22936, 22937, 22949, 23044, 23048, 23061,
        23066, 23072, 23077, 23078, 23081, 23109, 23112, 23113, 23121, 23125, 23126, 23128, 23129, 23138, 23141, 23144,
        23146, 23169, 23178, 23186, 23189, 23190, 23192, 23194, 23201, 24581, 24596, 24598, 24601, 24613, 24644, 24656,
        24661, 24662, 24664, 24666, 24673, 24676, 24678, 24681, 24705, 24726, 24741, 24833, 24836, 24838, 24841, 24850,
        24853, 24865, 24866, 24870, 24873, 24901, 24905, 24913, 24917, 24918, 24921, 24933, 24934, 24938, 24964, 24970,
        24978, 24981, 24993, 24998, 25001, 25105, 25110, 25113, 25152, 25153, 25158, 25173, 25174, 25176, 25184, 25221,
        25233, 25238, 25253, 25617, 25618, 25621, 25622, 25626, 25633, 25638, 25641, 25664, 25666, 25669, 25672, 25674,
        25681, 25684, 25685, 25686, 25689, 25690, 25696, 25698, 25701, 25732, 25733, 25737, 25744, 25746, 25748, 25749,
        25750, 25752, 25754, 25761, 25764, 25769, 25861, 25864, 25866, 25873, 25877, 25878, 25881, 25924, 25925, 25926,
        25929, 25936, 25937, 25940, 25941, 25942, 25945, 25953, 25956, 25957, 25958, 25961, 25990, 25993, 25994, 26001,
        26005, 26006, 26009, 26010, 26018, 26021, 26022, 26024, 26114, 26121, 26133, 26144, 26150, 26152, 26153, 26176,
        26181, 26184, 26186, 26193, 26196, 26197, 26198, 26200, 26202, 26208, 26213, 26216, 26240, 26242, 26245, 26250,
        26260, 26262, 26264, 26265, 26272, 26276, 26278, 26282, 26646, 26649, 26661, 26689, 26706, 26709, 26714, 26721,
        26729, 26757, 26769, 26776, 26790, 26881, 26884, 26896, 26901, 26913, 26916, 26918, 26921, 26944, 26945, 26949,
        26950, 26952, 26961, 26964, 26965, 26966, 26969, 26976, 26981, 26986, 27010, 27012, 27018, 27029, 27041, 27044,
        27045, 27049, 27153, 27158, 27160, 27201, 27204, 27209, 27216, 27221, 27224, 27226, 27236, 27237, 27241, 27270,
        27284, 27288, 27290, 27302, 32768, 32770, 32776, 32778, 32800, 32802, 32808, 32810, 32837, 32848, 32849, 32852,
        32854, 32857, 32869, 32896, 32898, 32904, 32906, 32917, 32928, 32930, 32936, 32938, 33029, 33041, 33044, 33046,
        33049, 33061, 33089, 33092, 33097, 33104, 33106, 33109, 33110, 33112, 33113, 33124, 33126, 33129, 33157, 33161,
        33172, 33174, 33177, 33189, 33280, 33282, 33288, 33290, 33301, 33312, 33314, 33320, 33322, 33361, 33364, 33369,
        33381, 33408, 33410, 33416, 33418, 33429, 33440, 33442, 33448, 33450, 33812, 33817, 33857, 33860, 33873, 33877,
        33882, 33889, 33892, 33897, 33940, 33945, 34049, 34057, 34066, 34069, 34074, 34086, 34089, 34112, 34113, 34117,
        34120, 34129, 34132, 34133, 34134, 34137, 34138, 34149, 34150, 34152, 34154, 34177, 34180, 34182, 34185, 34192,
        34194, 34197, 34200, 34214, 34321, 34326, 34329, 34341, 34369, 34372, 34377, 34378, 34384, 34389, 34393, 34394,
        34401, 34406, 34410, 34437, 34449, 34458, 34468, 34816, 34818, 34824, 34826, 34837, 34848, 34850, 34856, 34858,
        34881, 34885, 34897, 34900, 34905, 34917, 34921, 34944, 34946, 34952, 34954, 34965, 34976, 34978, 34984, 34986,
        35077, 35078, 35089, 35092, 35094, 35109, 35137, 35140, 35142, 35145, 35152, 35154, 35157, 35162, 35169, 35172,
        35205, 35222, 35225, 35237, 35328, 35330, 35336, 35338, 35349, 35360, 35362, 35368, 35370, 35397, 35409, 35412,
        35414, 35456, 35458, 35464, 35466, 35477, 35488, 35490, 35496, 35498, 36869, 36881, 36886, 36888, 36889, 36901,
        36929, 36934, 36937, 36949, 36952, 36954, 36969, 36970, 36997, 37009, 37012, 37014, 37017, 37029, 37121, 37124,
        37126, 37129, 37136, 37141, 37144, 37146, 37153, 37156, 37158, 37161, 37184, 37189, 37200, 37201, 37204, 37205,
        37206, 37209, 37218, 37221, 37252, 37254, 37266, 37269, 37272, 37281, 37284, 37286, 37289, 37381, 37393, 37396,
        37401, 37413, 37444, 37446, 37449, 37456, 37458, 37461, 37464, 37478, 37481, 37509, 37524, 37526, 37545, 37889,
        37892, 37894, 37904, 37909, 37912, 37926, 37952, 37962, 37969, 37972, 37973, 37974, 37976, 37977, 37984, 37985,
        37986, 37989, 38020, 38022, 38034, 38036, 38037, 38040, 38049, 38057, 38144, 38149, 38152, 38154, 38160, 38161,
        38164, 38165, 38166, 38169, 38177, 38181, 38185, 38186, 38209, 38212, 38213, 38214, 38217, 38224, 38225, 38226,
        38228, 38229, 38230, 38232, 38233, 38234, 38241, 38244, 38245, 38246, 38249, 38273, 38277, 38280, 38289, 38290,
        38292, 38293, 38294, 38297, 38298, 38304, 38306, 38309, 38312, 38314, 38401, 38404, 38416, 38421, 38425, 38432,
        38438, 38441, 38469, 38472, 38473, 38481, 38482, 38485, 38486, 38489, 38501, 38504, 38530, 38532, 38537, 38538,
        38546, 38548, 38549, 38564, 38566, 38569, 38917, 38934, 38937, 38949, 38977, 38982, 38992, 38994, 38997, 38998,
        39002, 39012, 39013, 39045, 39057, 39062, 39065, 39077, 39172, 39174, 39177, 39184, 39186, 39189, 39192, 39194,
        39200, 39201, 39204, 39206, 39232, 39234, 39237, 39240, 39242, 39249, 39252, 39253, 39254, 39257, 39266, 39269,
        39270, 39274, 39297, 39300, 39312, 39314, 39317, 39322, 39329, 39334, 39429, 39445, 39461, 39492, 39494, 39497,
        39504, 39509, 39512, 39521, 39557, 39569, 39572, 39573, 39574, 40960, 40962, 40968, 40970, 40981, 40992, 40994,
        41000, 41002, 41029, 41041, 41044, 41046, 41049, 41088, 41090, 41096, 41098, 41109, 41120, 41122, 41128, 41130,
        41221, 41225, 41233, 41236, 41238, 41241, 41242, 41286, 41289, 41297, 41301, 41304, 41306, 41313, 41316, 41349,
        41360, 41362, 41366, 41369, 41474, 41480, 41482, 41488, 41497, 41506, 41512, 41514, 41541, 41553, 41558, 41561,
        41573, 41600, 41602, 41608, 41610, 41621, 41632, 41634, 41640, 41642, 42009, 42021, 42049, 42052, 42064, 42068,
        42069, 42072, 42074, 42081, 42085, 42086, 42088, 42089, 42117, 42246, 42249, 42256, 42258, 42261, 42264, 42278,
        42281, 42306, 42309, 42321, 42324, 42325, 42326, 42329, 42341, 42346, 42369, 42372, 42373, 42374, 42377, 42386,
        42389, 42392, 42501, 42513, 42518, 42522, 42529, 42533, 42564, 42566, 42570, 42578, 42581, 42582, 42584, 42592,
        42594, 42630, 42640, 42645, 42646, 42649, 42657, 42660, 42662, 43008, 43010, 43016, 43018, 43040, 43042, 43048,
        43050, 43089, 43092, 43094, 43097, 43136, 43138, 43144, 43146, 43157, 43168, 43170, 43176, 43178, 43269, 43284,
        43289, 43297, 43301, 43329, 43344, 43349, 43354, 43361, 43366, 43369, 43408, 43414, 43520, 43522, 43528, 43530,
        43552, 43554, 43560, 43562, 43601, 43604, 43606, 43648, 43650, 43656, 43658, 43669, 43680, 43682, 43688, 43690,
    };
    static const uint16_t kgrid_2bit_1024[1024] = {
            0,     2,     5,     8,    10,    17,    20,    22,    25,    32,    34,    37,    40,    65,    68,    70,
           73,    80,    82,    85,    88,    97,   100,   102,   105,   128,   130,   133,   136,   145,   148,   160,
          165,   170,   257,   260,   262,   265,   272,   274,   277,   280,   289,   292,   320,   322,   325,   328,
          337,   340,   342,   345,   352,   357,   360,   385,   388,   400,   402,   405,   417,   420,   512,   514,
          517,   520,   529,   532,   544,   554,   577,   580,   582,   585,   592,   597,   640,   645,   650,   660,
          674,  1025,  1028,  1030,  1033,  1040,  1042,  1045,  1048,  1057,  1060,  1062,  1065,  1088,  1090,  1093,
         1096,  1098,  1105,  1108,  1110,  1113,  1120,  1122,  1125,  1153,  1156,  1158,  1161,  1168,  1173,  1176,
         1185,  1188,  1280,  1282,  1285,  1288,  1290,  1297,  1300,  1302,  1305,  1312,  1317,  1320,  1345,  1348,
         1350,  1353,  1360,  1362,  1365,  1368,  1377,  1380,  1408,  1410,  1413,  1416,  1425,  1428,  1440,  1537,
         1540,  1542,  1545,  1552,  1557,  1600,  1605,  1608,  1617,  1620,  1632,  1665,  1668,  1680,  2048,  2050,
         2053,  2056,  2065,  2068,  2070,  2073,  2080,  2085,  2090,  2113,  2116,  2118,  2121,  2128,  2130,  2133,
         2136,  2145,  2148,  2176,  2181,  2196,  2218,  2305,  2308,  2320,  2322,  2325,  2328,  2337,  2368,  2373,
         2376,  2385,  2388,  2400,  2433,  2448,  2560,  2577,  2580,  2594,  2600,  2602,  2640,  2713,  4097,  4100,
         4102,  4105,  4112,  4114,  4117,  4120,  4129,  4132,  4134,  4160,  4162,  4165,  4168,  4177,  4180,  4182,
         4185,  4192,  4194,  4197,  4200,  4225,  4228,  4230,  4240,  4245,  4248,  4257,  4260,  4352,  4354,  4357,
         4360,  4362,  4369,  4372,  4374,  4377,  4384,  4386,  4389,  4392,  4417,  4420,  4422,  4425,  4432,  4434,
         4437,  4440,  4449,  4452,  4480,  4482,  4485,  4488,  4497,  4500,  4609,  4612,  4617,  4624,  4629,  4641,
         4644,  4672,  4677,  4689,  4692,  4737,  4740,  4752,  5120,  5122,  5125,  5128,  5137,  5140,  5142,  5145,
         5152,  5157,  5160,  5185,  5188,  5190,  5193,  5200,  5202,  5205,  5208,  5217,  5220,  5248,  5250,  5253,
         5256,  5265,  5268,  5280,  5377,  5380,  5382,  5385,  5392,  5394,  5397,  5400,  5409,  5412,  5440,  5442,
         5445,  5448,  5457,  5460,  5472,  5505,  5508,  5520,  5632,  5637,  5640,  5649,  5652,  5664,  5697,  5700,
         5712,  5760,  5802,  6145,  6148,  6150,  6153,  6160,  6165,  6168,  6177,  6208,  6210,  6213,  6216,  6225,
         6228,  6240,  6273,  6276,  6400,  6402,  6405,  6408,  6417,  6420,  6432,  6465,  6468,  6480,  6505,  6562,
         6660,  6672,  6720,  6742,  8192,  8194,  8197,  8200,  8209,  8212,  8214,  8217,  8224,  8229,  8234,  8257,
         8260,  8272,  8274,  8277,  8292,  8320,  8330,  8340,  8362,  8449,  8452,  8464,  8466,  8469,  8481,  8512,
         8514,  8517,  8529,  8532,  8544,  8577,  8580,  8592,  8704,  8714,  8738,  8744,  8746,  8772,  8784,  8840,
         8842,  8872,  9217,  9220,  9222,  9225,  9232,  9237,  9240,  9249,  9252,  9280,  9282,  9285,  9288,  9297,
         9300,  9312,  9345,  9348,  9360,  9472,  9477,  9480,  9489,  9492,  9504,  9537,  9540,  9552,  9574,  9600,
         9729,  9732,  9744,  9792,  9817, 10240, 10245, 10257, 10260, 10305, 10308, 10320, 10378, 10410, 10497, 10500,
        10512, 10645, 10762, 10786, 10852, 10888, 10890, 16385, 16388, 16390, 16393, 16400, 16402, 16405, 16408, 16410,
        16417, 16420, 16422, 16448, 16450, 16453, 16456, 16458, 16465, 16468, 16470, 16473, 16480, 16482, 16485, 16513,
        16516, 16528, 16533, 16536, 16545, 16548, 16640, 16642, 16645, 16648, 16657, 16660, 16662, 16665, 16672, 16674,
        16677, 16705, 16708, 16710, 16713, 16720, 16722, 16725, 16728, 16737, 16740, 16768, 16770, 16773, 16776, 16785,
        16788, 16800, 16897, 16900, 16912, 16914, 16917, 16920, 16932, 16960, 16965, 16968, 16977, 16980, 16992, 17025,
        17028, 17408, 17410, 17413, 17416, 17418, 17425, 17428, 17430, 17433, 17440, 17442, 17445, 17448, 17473, 17476,
        17478, 17481, 17488, 17490, 17493, 17496, 17505, 17508, 17536, 17538, 17541, 17544, 17553, 17556, 17568, 17665,
        17668, 17670, 17673, 17680, 17682, 17685, 17688, 17697, 17700, 17728, 17730, 17733, 17736, 17745, 17748, 17760,
        17770, 17793, 17796, 17808, 17920, 17922, 17925, 17928, 17937, 17940, 17952, 17985, 17988, 18000, 18048, 18085,
        18433, 18436, 18441, 18448, 18450, 18453, 18456, 18465, 18468, 18496, 18498, 18501, 18504, 18513, 18516, 18528,
        18564, 18576, 18688, 18690, 18693, 18696, 18705, 18708, 18720, 18753, 18756, 18768, 18816, 18838, 18945, 18948,
        18960, 19008, 20480, 20482, 20485, 20488, 20497, 20500, 20502, 20505, 20512, 20514, 20517, 20520, 20545, 20548,
        20550, 20553, 20560, 20562, 20565, 20568, 20577, 20580, 20608, 20610, 20613, 20616, 20625, 20628, 20737, 20740,
        20742, 20745, 20752, 20754, 20757, 20760, 20769, 20772, 20800, 20802, 20805, 20808, 20817, 20820, 20832, 20865,
        20868, 20880, 20992, 20997, 21000, 21009, 21012, 21024, 21057, 21060, 21072, 21097, 21120, 21505, 21508, 21510,
        21513, 21520, 21522, 21525, 21528, 21537, 21540, 21568, 21570, 21573, 21576, 21585, 21588, 21600, 21633, 21636,
        21648, 21760, 21762, 21765, 21768, 21777, 21780, 21792, 21825, 21828, 21840, 21888, 22017, 22020, 22032, 22054,
        22080, 22528, 22530, 22533, 22536, 22545, 22548, 22560, 22593, 22596, 22608, 22618, 22656, 22785, 22788, 22800,
        22848, 23040, 23065, 23173, 23208, 24577, 24580, 24582, 24592, 24594, 24597, 24600, 24609, 24612, 24640, 24645,
        24648, 24657, 24660, 24672, 24708, 24720, 24832, 24834, 24837, 24840, 24849, 24852, 24864, 24897, 24900, 24912,
        24960, 24985, 25092, 25104, 25152, 25174, 25249, 25600, 25605, 25608, 25617, 25620, 25632, 25665, 25668, 25680,
        25728, 25857, 25860, 25872, 25920, 25930, 25960, 26002, 26112, 26260, 26625, 26628, 26640, 26725, 26776, 26880,
        26922, 27202, 27297, 32768, 32770, 32773, 32776, 32785, 32788, 32793, 32800, 32805, 32833, 32836, 32848, 32850,
        32853, 32856, 32865, 32896, 32901, 32913, 32916, 33025, 33028, 33033, 33040, 33042, 33045, 33048, 33057, 33060,
        33088, 33090, 33093, 33096, 33105, 33108, 33153, 33156, 33168, 33193, 33280, 33285, 33290, 33297, 33300, 33345,
        33348, 33360, 33793, 33796, 33798, 33801, 33808, 33810, 33813, 33816, 33825, 33856, 33858, 33861, 33864, 33873,
        33876, 33888, 33921, 33924, 33936, 34048, 34050, 34053, 34056, 34065, 34068, 34080, 34113, 34116, 34128, 34176,
        34186, 34305, 34308, 34320, 34345, 34368, 34816, 34821, 34833, 34836, 34881, 34884, 34896, 34978, 35073, 35076,
        35136, 35173, 35362, 35416, 35418, 35458, 35490, 36865, 36868, 36873, 36880, 36882, 36885, 36888, 36900, 36928,
        36930, 36933, 36936, 36945, 36948, 36960, 36993, 36996, 37008, 37120, 37125, 37137, 37140, 37185, 37188, 37200,
        37210, 37377, 37380, 37392, 37440, 37542, 37888, 37890, 37893, 37896, 37905, 37908, 37920, 37953, 37956, 37968,
        38016, 38038, 38145, 38148, 38160, 38208, 38296, 38305, 38400, 38470, 38500, 38913, 38916, 38928, 38950, 38976,
        39081, 39168, 39241, 39250, 39568, 40960, 40965, 40970, 40980, 40994, 41002, 41025, 41028, 41040, 41122, 41130,
        41280, 41317, 41474, 41482, 41506, 41512, 41514, 41602, 41608, 41610, 41640, 41985, 41988, 42000, 42048, 42121,
        42148, 42240, 42265, 42577, 43018, 43048, 43170, 43348, 43398, 43528, 43530, 43552, 43554, 43560, 43656, 43690,
    };

    static const int8_t kvalues_iq2[3] = { 0x08, 0x19, 0x2b };
    static const int8_t kvalues_iq1[3] = { -8, 0, 8 };

    // alternatively, this could be 0xAAAA = 43690, but that would be much bigger unnecessarily.
    const int kmap_size = pow3[8]; // 3**8 = 6561
    const int nwant = type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M ? 3 : type == GGML_TYPE_IQ2_S ? 1 : 2;
    const uint16_t * kgrid = type == GGML_TYPE_IQ2_XXS ? kgrid_2bit_256 :
                             type == GGML_TYPE_IQ2_XS  ? kgrid_2bit_512 :
                             type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M ? kgrid_1bit_2048 : kgrid_2bit_1024;
    const int8_t * kvalues = type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M ? kvalues_iq1 : kvalues_iq2;

    uint64_t * kgrid_q2xs;
    int      * kmap_q2xs;
    uint16_t * kneighbours_q2xs;

    //printf("================================================================= %s(grid_size = %d)\n", __func__, grid_size);

    kmap_q2xs = (int *) malloc(kmap_size * sizeof(int));
    for (int i = 0; i < kmap_size; ++i) {
        kmap_q2xs[i] = -1;
    }
    for (int i = 0; i < grid_size; ++i) {
        int index = 0;
        for (int k = 0; k < 8; ++k) {
            int l = (kgrid[i] >> (2 * k)) & 0x3;
            if (l >= 3) {
                fprintf(stderr, "%s: %i is an out-of-bounds grid value at kgrid[%i] >> 2*%i\n", __func__, l, i, k);
                GGML_ASSERT(false);
            }
            index += l * pow3[k];
        }
        kmap_q2xs[index] = i;
    }
    iq2_data[gindex].map = kmap_q2xs;

    kgrid_q2xs = (uint64_t *) malloc(grid_size * sizeof(uint64_t));
    for (int i = 0; i < grid_size; ++i) {
        int8_t * pg = (int8_t *)(kgrid_q2xs + i);
        for (int k = 0; k < 8; ++k) {
            const int l = (kgrid[i] >> (2 * k)) & 3;
            pg[k] = kvalues[l];
        }
    }
    iq2_data[gindex].grid = kgrid_q2xs;

    // TODO: figure out which neighbours can be omitted
    // TODO: figure out how to get all points corresponding to the adjacent voronoi hyperplanes
    int8_t pos[8];
    int * dist2 = (int *)malloc(2*grid_size*sizeof(int));
    int num_neighbors = 0;
    int num_not_in_map = 0;
    for (int i = 0; i < kmap_size; ++i) {
        if (kmap_q2xs[i] >= 0) { continue; }
        ++num_not_in_map;
        for (int k = 0; k < 8; ++k) {
            int l = (i / pow3[k]) % 3;
            pos[k] = kvalues[l];
        }
        for (int j = 0; j < grid_size; ++j) {
            const int8_t * pg = (const int8_t *)(kgrid_q2xs + j);
            int d2 = 0;
            for (int k = 0; k < 8; ++k) { d2 += (pg[k] - pos[k])*(pg[k] - pos[k]); }
            dist2[2*j+0] = d2;
            dist2[2*j+1] = j;
        }
        qsort(dist2, grid_size, 2*sizeof(int), iq2_compare_func);
        int n = 0; int d2 = dist2[0];
        int nhave = 1;
        for (int j = 0; j < grid_size; ++j) {
            if (dist2[2*j] > d2) {
                if (nhave == nwant) { break; }
                d2 = dist2[2*j];
                ++nhave;
            }
            ++n;
        }
        num_neighbors += n;
    }
    printf("%s: %d neighbours in total\n", __func__, num_neighbors);
    kneighbours_q2xs = (uint16_t *)malloc((num_neighbors + num_not_in_map)*sizeof(uint16_t));
    iq2_data[gindex].neighbours = kneighbours_q2xs;
    int counter = 0;
    for (int i = 0; i < kmap_size; ++i) {
        if (kmap_q2xs[i] >= 0) { continue; }
        for (int k = 0; k < 8; ++k) {
            int l = (i / pow3[k]) % 3;
            pos[k] = kvalues[l];
        }
        for (int j = 0; j < grid_size; ++j) {
            const int8_t * pg = (const int8_t *)(kgrid_q2xs + j);
            int d2 = 0;
            for (int k = 0; k < 8; ++k) { d2 += (pg[k] - pos[k])*(pg[k] - pos[k]); }
            dist2[2*j+0] = d2;
            dist2[2*j+1] = j;
        }
        qsort(dist2, grid_size, 2*sizeof(int), iq2_compare_func);
        kmap_q2xs[i] = -(counter + 1);
        int d2 = dist2[0];
        uint16_t * start = &kneighbours_q2xs[counter++];
        int n = 0;
        int nhave = 1;
        for (int j = 0; j < grid_size; ++j) {
            if (dist2[2*j] > d2) {
                if (nhave == nwant) { break; }
                d2 = dist2[2*j];
                ++nhave;
            }
            kneighbours_q2xs[counter++] = dist2[2*j+1];
            ++n;
        }
        *start = n;
    }
    GGML_ASSERT(counter == num_neighbors + num_not_in_map);
    free(dist2);
}

void iq2xs_free_impl(enum ggml_type type) {
    GGML_ASSERT(type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M || type == GGML_TYPE_IQ2_S);
    const int gindex = iq2_data_index(type);
    if (iq2_data[gindex].map) {
        free(iq2_data[gindex].grid);       iq2_data[gindex].grid = NULL;
        free(iq2_data[gindex].map);        iq2_data[gindex].map  = NULL;
        free(iq2_data[gindex].neighbours); iq2_data[gindex].neighbours = NULL;
    }
}

// returns grid_index
// and stores relative sumqx and sumq2 in the respective arguments
static int iq2_find_relative_neighbour(const struct k_sort * GGML_RESTRICT k_sort,
                                       const uint64_t      * GGML_RESTRICT kgrid,
                                       const int           * GGML_RESTRICT kmap,
                                       const uint16_t      * GGML_RESTRICT kneighbours,
                                       const float         * GGML_RESTRICT xval,
                                       const float         * GGML_RESTRICT weight,
                                       const int8_t        * GGML_RESTRICT L,
                                       float               * GGML_RESTRICT sumqx,
                                       float               * GGML_RESTRICT sumq2,
                                       int                                 grid_offset) {
    const int pow3[9] = { 1, 3, 9, 27, 81, 243, 729, 2187, 6561 };
    int index = 0;
    int8_t p[8];
    for (int k = 0; k < 8; ++k) {
        const int8_t l = L[k] + k_sort->mid_k;
        GGML_ASSERT(l >= 0 && l < 3);
        index += l * pow3[k];
        p[k] = k_sort->k_values[l];
    }
    int grid_index = kmap[index];

    float sumqx_new = 0.0f;
    float sumq2_new = 0.0f;
    float best_d2 = FLT_MAX;

    if (grid_index < 0) {
        const uint16_t * neighbours = kneighbours - (grid_index + 1);
        const int num_neighbours = neighbours[0];

        float prev_sumqx = 0.0f;
        float prev_sumq2 = 0.0f;
        float waux[8];
        for (int k = 0; k < 8; ++k) {
            prev_sumqx += weight[k] * (xval[k] * p[k]);
            prev_sumq2 += weight[k] * (p[k] * p[k]);
            waux[k] = sqrtf(weight[k]);
        }
        const float prev_scale = prev_sumq2 > 0.0f ? prev_sumqx / prev_sumq2 : 0.0f;

        for (int i = 1; i <= num_neighbours; ++i) {
            const int8_t * pg = (const int8_t *)(kgrid + neighbours[i]);
            float d2 = 0.0f;
            for (int k = 0; k < 8; ++k) {
                const float diff = prev_scale * (grid_offset + pg[k]) - xval[k];
                d2 += waux[k] * diff * diff;
            }

            if (d2 < best_d2) {
                best_d2 = d2;
                grid_index = neighbours[i];
            }
        }

        if (grid_index >= 0) {
            const int8_t * pg = (const int8_t *)(kgrid + grid_index);
            for (int k = 0; k < 8; ++k) {
                const float odd = (grid_offset + pg[k]) + p[k];
                const float step = (grid_offset + pg[k]) - p[k];
                sumqx_new += weight[k] * (xval[k] * step);
                sumq2_new += weight[k] * (odd * step);
            }
        }
    }

    *sumqx = sumqx_new;
    *sumq2 = sumq2_new;

    return grid_index;
}

static int iq2_find_best_neighbour(const uint16_t * GGML_RESTRICT neighbours, const uint64_t * GGML_RESTRICT grid,
        const float * GGML_RESTRICT xval, const float * GGML_RESTRICT weight, float scale, int8_t * GGML_RESTRICT L) {
    int num_neighbors = neighbours[0];
    GGML_ASSERT(num_neighbors > 0);
    float best_d2 = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 8; ++i) {
            float q = pg[i];
            float diff = scale*q - xval[i];
            d2 += weight[i]*diff*diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2; grid_index = neighbours[j];
        }
    }
    GGML_ASSERT(grid_index >= 0);
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 8; ++i) L[i] = (pg[i] - 1)/2;
    return grid_index;
}

static float make_iq2_quants(int n, struct k_sort * k_sort, const uint64_t * grid, const int * kmap, const uint16_t * kneighbours, const float * xval, const float * weight, int8_t * Laux, float * sumqx_aux, float * sumq2_aux, int * grid_idx_aux, int * grid_idx) {
    GGML_ASSERT(n % 8 == 0);
    const int n_idx = n / 8;

    for (int i = 0; i < n_idx; ++i) {
        sumqx_aux[i] = 0.0f;
        sumq2_aux[i] = 0.0f;
        grid_idx_aux[i] = 0;
        grid_idx[i] = 0;
    }

    float sumqx = 0.0f;
    float sumq2 = 0.0f;
    float max = 0.0f;
    float w_amax = -1.0f;
    int w_amax_i = -1;
    const float kmin = k_sort->kmin;
    for (int i = 0; i < n; ++i) {
        const float x = xval[i];
        const float w = weight[i];
        const float w_ax = fabsf(x) * w;
        max = MAX(x, max);
        if (w_ax > w_amax) {
            w_amax = w_ax;
            w_amax_i = i;
        }
        Laux[i] = k_sort->mid_k;
        sumqx += w * (x * kmin);
        sumq2 += w * (kmin * kmin);
    }
    if (max < GROUP_MAX_EPS_IQ2_S) {
        return 0.0f;
    }

    k_sort_set_x_L(k_sort, n, w_amax_i, xval, Laux, false);

    float best;
    float best_sumqx;
    float best_sumq2;
    if (sumq2 > 0.0f) {
        best = sumqx * sumqx;
        best_sumqx = sumqx;
        best_sumq2 = sumq2;
    } else {
        best = 0.0f;
        best_sumqx = 0.0f;
        best_sumq2 = 1.0f;
    }
    for (int i = 0; i < k_sort->n; ++i) {
        const int ii = k_sort->ids[i];
        const int k_i = k_sort->k_ids[i];
        const float odd = k_sort->odd[k_i];
        const float step = k_sort->step[k_i];
        const float w = weight[ii];
        const int g_i = ii / 8;
        sumqx += w * (xval[ii] * step);
        sumq2 += w * (odd * step);
        Laux[ii] = k_i;

        const int grid_index = iq2_find_relative_neighbour(k_sort, grid, kmap, kneighbours, xval + 8*g_i, weight + 8*g_i, Laux + 8*g_i, sumqx_aux + g_i, sumq2_aux + g_i, 0);

        if (grid_index == grid_idx_aux[g_i]) { continue; }
        if (grid_index < 0) { break; }

        grid_idx_aux[g_i] = grid_index;

        // avoid some subtraction numerical instabilities by having relative sumqx and sumq2 per grid index
        float sumqx_cur = sumqx;
        float sumq2_cur = sumq2;
        for (int j = 0; j < n_idx; ++j) {
            sumqx_cur += sumqx_aux[j];
            sumq2_cur += sumq2_aux[j];
        }

        const float current = sumqx_cur * sumqx_cur;
        if (sumq2_cur > 0.0f && current * best_sumq2 > best * sumq2_cur) {
            best = current;
            best_sumqx = sumqx_cur;
            best_sumq2 = sumq2_cur;
            for (int j = 0; j < n_idx; ++j) {
                grid_idx[j] = grid_idx_aux[j];
            }
        }
    }

    return best_sumq2 > 0.0f ? best_sumqx / best_sumq2 : 0.0f;
}

static void quantize_row_iq2_xxs_impl(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t n, const float * GGML_RESTRICT quant_weights) {

    const int gindex = iq2_data_index(GGML_TYPE_IQ2_XXS);

    const uint64_t * kgrid_q2xs       = iq2_data[gindex].grid;
    const int      * kmap_q2xs        = iq2_data[gindex].map;
    const uint16_t * kneighbours_q2xs = iq2_data[gindex].neighbours;

    GGML_ASSERT(quant_weights    && "missing quantization weights");
    GGML_ASSERT(kgrid_q2xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kmap_q2xs        && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbours_q2xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    const int8_t k_values_iq2xxs[3] = { 0x08, 0x19, 0x2b };
    // the quantized scales are in {0.125, 0.375, 0.625, ... }
    // which are the odd numbers divided by 8
    const int8_t k_values_iq2xxs_s[16] = { 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31 };

    const int64_t nbl = n/QK_K;

    block_iq2_xxs * y = vy;

    float scales[QK_K/32];
    float sw[QK_K/32];
    int8_t Ls[QK_K/32];
    int8_t Lsaux[QK_K/32];
    float weight[32];
    float xval[32];
    int8_t Laux[32];
    float sumqx_aux[4];
    float sumq2_aux[4];
    int grid_idx[4];
    int grid_idx_aux[4];
    uint8_t block_signs[4];
    uint32_t q2[2*(QK_K/32)];
    struct k_sort k_sort;
    uint8_t buf[K_SORT_BUF_SIZE_NL(32, 3, 2)];
    struct k_sort k_sort_s;
    uint8_t buf_s[K_SORT_BUF_SIZE_NL(QK_K/32, 16, 15)];

    k_sort_init(&k_sort, 32, 3, k_values_iq2xxs, buf);
    k_sort_init(&k_sort_s, QK_K/32, 16, k_values_iq2xxs_s, buf_s);

    for (int ibl = 0; ibl < nbl; ++ibl) {

        y[ibl].d = GGML_FP32_TO_FP16(0.f);
        memset(q2, 0, QK_K/4);

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) {
            sumx2 += xbl[i]*xbl[i];
        }
        float sigma2 = 2*sumx2/QK_K;

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const float * xb = xbl + 32*ib;
            const float * qw = quant_weights + QK_K*ibl + 32*ib;
            for (int i = 0; i < 32; ++i) {
                weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            }
            {
                float sumw = 0.0f;
                for (int i = 0; i < 32; ++i) {
                    sumw += weight[i];
                }
                sw[ib] = sumw;
            }
            for (int k = 0; k < 4; ++k) {
                int nflip = 0;
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) {
                        xval[8*k + i] = xb[8*k + i];
                    } else {
                        xval[8*k + i] = -xb[8*k + i];
                        ++nflip;
                        s |= (1 << i);
                    }
                }
                if (nflip%2) {
                    // uneven parity of flips can't be directly stored
                    // the least significant weight has its sign flipped
                    int imin = 0;
                    float min = weight[8*k + imin] * fabsf(xb[8*k + imin]);
                    for (int i = 1; i < 8; ++i) {
                        float ax = weight[8*k + i] * fabsf(xb[8*k + i]);
                        if (ax < min) {
                            min = ax;
                            imin = i;
                        }
                    }
                    xval[8*k+imin] = -xval[8*k+imin];
                    s ^= (1 << imin);
                }
                block_signs[k] = s & 127;
            }

            const float scale = make_iq2_quants(32, &k_sort, kgrid_q2xs, kmap_q2xs, kneighbours_q2xs, xval, weight, Laux, sumqx_aux, sumq2_aux, grid_idx_aux, grid_idx);

            for (int k = 0; k < 4; ++k) {
                int grid_index = grid_idx[k];
                q2[2*ib+0] |= ((uint32_t) grid_index << 8*k);
                q2[2*ib+1] |= (block_signs[k] << 7*k);
            }
            GGML_ASSERT(scale >= 0);
            scales[ib] = scale;
        }

        const float d = make_qkxs_nl_quants(QK_K/32, scales, sw, Ls, Lsaux, &k_sort_s, false, true);
        y[ibl].d = GGML_FP32_TO_FP16(d * 8.0f);
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const uint8_t l = Ls[ib];
            q2[2*ib+1] |= ((uint32_t)l << 28);
        }
        memcpy(y[ibl].qs, q2, QK_K/4);
    }
}

static void quantize_row_iq2_xs_impl(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t n, const float * GGML_RESTRICT quant_weights) {

    const int gindex = iq2_data_index(GGML_TYPE_IQ2_XS);

    const uint64_t * kgrid_q2xs       = iq2_data[gindex].grid;
    const int      * kmap_q2xs        = iq2_data[gindex].map;
    const uint16_t * kneighbours_q2xs = iq2_data[gindex].neighbours;

    GGML_ASSERT(quant_weights    && "missing quantization weights");
    GGML_ASSERT(kmap_q2xs        && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kgrid_q2xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbours_q2xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    const int8_t k_values_iq2xs[3] = { 0x08, 0x19, 0x2b };
    // the quantized scales are in {0.125, 0.375, 0.625, ... }
    // which are the odd numbers divided by 8
    const int8_t k_values_iq2xs_s[16] = { 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31 };

    const int64_t nbl = n/QK_K;

    block_iq2_xs * y = vy;

    float scales[QK_K/16];
    float sw[QK_K/16];
    int8_t Ls[QK_K/16];
    int8_t Lsaux[QK_K/16];
    float weight[16];
    float xval[16];
    int8_t Laux[16];
    int grid_idx[2];
    int grid_idx_aux[2];
    float sumqx_aux[2];
    float sumq2_aux[2];
    uint8_t block_signs[2];
    uint16_t q2[2*(QK_K/16)];
    struct k_sort k_sort;
    uint8_t buf[K_SORT_BUF_SIZE_NL(16, 3, 2)];
    struct k_sort k_sort_s;
    uint8_t buf_s[K_SORT_BUF_SIZE_NL(QK_K/16, 16, 15)];

    k_sort_init(&k_sort, 16, 3, k_values_iq2xs, buf);
    k_sort_init(&k_sort_s, QK_K/16, 16, k_values_iq2xs_s, buf_s);

    for (int ibl = 0; ibl < nbl; ++ibl) {

        y[ibl].d = GGML_FP32_TO_FP16(0.f);
        memset(q2, 0, QK_K/4);
        memset(y[ibl].scales, 0, QK_K/32);

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) {
            sumx2 += xbl[i]*xbl[i];
        }
        float sigma2 = 2*sumx2/QK_K;

        for (int ib = 0; ib < QK_K/16; ++ib) {
            const float * xb = xbl + 16*ib;
            const float * qw = quant_weights + QK_K*ibl + 16*ib;
            for (int i = 0; i < 16; ++i) {
                weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            }
            {
                float sumw = 0.0f;
                for (int i = 0; i < 16; ++i) {
                    sumw += weight[i];
                }
                sw[ib] = sumw;
            }
            for (int k = 0; k < 2; ++k) {
                int nflip = 0;
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) {
                        xval[8*k + i] = xb[8*k + i];
                    } else {
                        xval[8*k + i] = -xb[8*k + i];
                        ++nflip;
                        s |= (1 << i);
                    }
                }
                if (nflip % 2) {
                    // uneven parity of flips can't be directly stored
                    // the least significant weight has its sign flipped
                    int imin = 0;
                    float min = weight[8*k + imin] * fabsf(xb[8*k + imin]);
                    for (int i = 1; i < 8; ++i) {
                        float ax = weight[8*k + i] * fabsf(xb[8*k + i]);
                        if (ax < min) {
                            min = ax;
                            imin = i;
                        }
                    }
                    xval[8*k+imin] = -xval[8*k+imin];
                    s ^= (1 << imin);
                }
                block_signs[k] = s & 127;
            }
            const float scale = make_iq2_quants(16, &k_sort, kgrid_q2xs, kmap_q2xs, kneighbours_q2xs, xval, weight, Laux, sumqx_aux, sumq2_aux, grid_idx_aux, grid_idx);
            for (int k = 0; k < 2; ++k) {
                const int grid_index = grid_idx[k];
                q2[2*ib+k] = grid_index | (block_signs[k] << 9);
            }
            GGML_ASSERT(scale >= 0.0f);
            scales[ib] = scale;
        }

        const float d = make_qkxs_nl_quants(QK_K/16, scales, sw, Ls, Lsaux, &k_sort_s, false, true);
        y[ibl].d = GGML_FP32_TO_FP16(d * 8.0f);
        for (int ib = 0; ib < QK_K/16; ++ib) {
            const uint8_t l = Ls[ib];
            if (ib % 2 == 0) {
                y[ibl].scales[ib / 2] = l;
            } else {
                y[ibl].scales[ib / 2] |= (l << 4);
            }
        }
        memcpy(y[ibl].qs, q2, QK_K/4);

    }
}

size_t quantize_iq2_xxs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    int64_t nblock = n_per_row/QK_K;
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq2_xxs_impl(src, qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += nblock*sizeof(block_iq2_xxs);
    }
    return nrow * nblock * sizeof(block_iq2_xxs);
}

size_t quantize_iq2_xs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    int64_t nblock = n_per_row/QK_K;
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq2_xs_impl(src, qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += nblock*sizeof(block_iq2_xs);
    }
    return nrow * nblock * sizeof(block_iq2_xs);
}

//
// ============================================= 3-bit using D4 lattice
//

typedef struct {
    uint32_t * grid;
    int      * map;
    uint16_t * neighbours;
} iq3_entry_t;

// IDEA: sort by scale and prune by distance
// then find boundaries and merge sort that with the other block subsets
// then compare for every candidate scale
typedef struct {
    float scale; // sumqx / sumq2
    // FIXME: is this really the correct formula???
    float d2; // diff2 = d2 + sumq2*(new_scale - scale)**2
    float sumqx;
    float sumq2;
    int16_t index;
    int16_t g_i;
} iqx_diff_t;

static iq3_entry_t iq3_data[2] = {
    {NULL, NULL, NULL},
    {NULL, NULL, NULL},
};

static inline int iq3_data_index(int grid_size) {
    (void)grid_size;
    GGML_ASSERT(grid_size == 256 || grid_size == 512);
    return grid_size == 256 ? 0 : 1;
}

static int iq3_compare_func(const void * left, const void * right) {
    const int * l = (const int *)left;
    const int * r = (const int *)right;
    return l[0] < r[0] ? -1 : l[0] > r[0] ? 1 : l[1] < r[1] ? -1 : l[1] > r[1] ? 1 : 0;
}

static int iqx_diff_scale_compare_asc(const void * left, const void * right) {
    const float l = ((const iqx_diff_t *) left)->scale;
    const float r = ((const iqx_diff_t *) right)->scale;

    if (l == r) { return 0; }
    return l < r ? -1 : 1;
}

void iq3xs_init_impl(int grid_size) {
    const int gindex = iq3_data_index(grid_size);
    if (iq3_data[gindex].grid) {
        return;
    }
    static const uint16_t kgrid_256[256] = {
            0,     2,     4,     9,    11,    15,    16,    18,    25,    34,    59,    61,    65,    67,    72,    74,
           81,    85,    88,    90,    97,   108,   120,   128,   130,   132,   137,   144,   146,   153,   155,   159,
          169,   175,   189,   193,   199,   200,   202,   213,   248,   267,   287,   292,   303,   315,   317,   321,
          327,   346,   362,   413,   436,   456,   460,   462,   483,   497,   513,   515,   520,   522,   529,   531,
          536,   538,   540,   551,   552,   576,   578,   585,   592,   594,   641,   643,   648,   650,   657,   664,
          698,   704,   706,   720,   729,   742,   758,   769,   773,   808,   848,   852,   870,   889,   901,   978,
          992,  1024,  1026,  1033,  1035,  1040,  1042,  1046,  1049,  1058,  1089,  1091,  1093,  1096,  1098,  1105,
         1112,  1139,  1143,  1144,  1152,  1154,  1161,  1167,  1168,  1170,  1183,  1184,  1197,  1217,  1224,  1228,
         1272,  1276,  1309,  1323,  1347,  1367,  1377,  1404,  1473,  1475,  1486,  1509,  1537,  1544,  1546,  1553,
         1555,  1576,  1589,  1594,  1600,  1602,  1616,  1625,  1636,  1638,  1665,  1667,  1672,  1685,  1706,  1722,
         1737,  1755,  1816,  1831,  1850,  1856,  1862,  1874,  1901,  1932,  1950,  1971,  2011,  2032,  2052,  2063,
         2077,  2079,  2091,  2095,  2172,  2192,  2207,  2208,  2224,  2230,  2247,  2277,  2308,  2345,  2356,  2389,
         2403,  2424,  2501,  2504,  2506,  2520,  2570,  2593,  2616,  2624,  2630,  2646,  2669,  2700,  2714,  2746,
         2754,  2795,  2824,  2835,  2839,  2874,  2882,  2905,  2984,  3028,  3042,  3092,  3108,  3110,  3124,  3153,
         3185,  3215,  3252,  3288,  3294,  3364,  3397,  3434,  3483,  3523,  3537,  3587,  3589,  3591,  3592,  3610,
         3626,  3670,  3680,  3722,  3749,  3754,  3776,  3789,  3803,  3824,  3857,  3873,  3904,  3906,  3924,  3992,
    };
    static const uint16_t kgrid_512[512] = {
            0,     1,     2,     5,     7,     8,     9,    10,    12,    14,    16,    17,    21,    27,    32,    34,
           37,    39,    41,    43,    48,    50,    57,    60,    63,    64,    65,    66,    68,    72,    73,    77,
           80,    83,    87,    89,    93,   100,   113,   117,   122,   128,   129,   133,   135,   136,   139,   142,
          145,   149,   152,   156,   162,   165,   167,   169,   171,   184,   187,   195,   201,   205,   208,   210,
          217,   219,   222,   228,   232,   234,   247,   249,   253,   256,   267,   271,   273,   276,   282,   288,
          291,   297,   312,   322,   324,   336,   338,   342,   347,   353,   357,   359,   374,   379,   390,   393,
          395,   409,   426,   441,   448,   450,   452,   464,   466,   470,   475,   488,   492,   512,   513,   514,
          516,   520,   521,   523,   525,   527,   528,   530,   537,   540,   542,   556,   558,   561,   570,   576,
          577,   579,   582,   584,   588,   593,   600,   603,   609,   616,   618,   632,   638,   640,   650,   653,
          655,   656,   660,   666,   672,   675,   685,   688,   698,   705,   708,   711,   712,   715,   721,   727,
          728,   732,   737,   754,   760,   771,   773,   778,   780,   793,   795,   802,   806,   808,   812,   833,
          840,   843,   849,   856,   858,   873,   912,   916,   919,   932,   934,   961,   963,   968,   970,   977,
          989,   993,  1010,  1016,  1024,  1025,  1027,  1029,  1031,  1032,  1034,  1036,  1038,  1041,  1043,  1047,
         1048,  1050,  1057,  1059,  1061,  1064,  1066,  1079,  1080,  1083,  1085,  1088,  1090,  1096,  1099,  1103,
         1106,  1109,  1113,  1116,  1122,  1129,  1153,  1156,  1159,  1169,  1171,  1176,  1183,  1185,  1195,  1199,
         1209,  1212,  1216,  1218,  1221,  1225,  1234,  1236,  1241,  1243,  1250,  1256,  1270,  1281,  1287,  1296,
         1299,  1306,  1309,  1313,  1338,  1341,  1348,  1353,  1362,  1375,  1376,  1387,  1400,  1408,  1410,  1415,
         1425,  1453,  1457,  1477,  1481,  1494,  1496,  1507,  1512,  1538,  1545,  1547,  1549,  1551,  1554,  1561,
         1563,  1565,  1570,  1572,  1575,  1577,  1587,  1593,  1601,  1603,  1605,  1612,  1617,  1619,  1632,  1648,
         1658,  1662,  1664,  1674,  1680,  1690,  1692,  1704,  1729,  1736,  1740,  1745,  1747,  1751,  1752,  1761,
         1763,  1767,  1773,  1787,  1795,  1801,  1806,  1810,  1817,  1834,  1840,  1844,  1857,  1864,  1866,  1877,
         1882,  1892,  1902,  1915,  1934,  1953,  1985,  1987,  2000,  2002,  2013,  2048,  2052,  2058,  2064,  2068,
         2071,  2074,  2081,  2088,  2104,  2114,  2119,  2121,  2123,  2130,  2136,  2141,  2147,  2153,  2157,  2177,
         2179,  2184,  2189,  2193,  2203,  2208,  2223,  2226,  2232,  2244,  2249,  2251,  2256,  2258,  2265,  2269,
         2304,  2306,  2324,  2335,  2336,  2361,  2373,  2375,  2385,  2418,  2443,  2460,  2480,  2504,  2509,  2520,
         2531,  2537,  2562,  2568,  2572,  2578,  2592,  2596,  2599,  2602,  2614,  2620,  2625,  2627,  2629,  2634,
         2641,  2650,  2682,  2688,  2697,  2707,  2712,  2718,  2731,  2754,  2759,  2760,  2775,  2788,  2793,  2805,
         2811,  2817,  2820,  2832,  2842,  2854,  2890,  2902,  2921,  2923,  2978,  3010,  3012,  3026,  3081,  3083,
         3085,  3097,  3099,  3120,  3136,  3152,  3159,  3188,  3210,  3228,  3234,  3245,  3250,  3256,  3264,  3276,
         3281,  3296,  3349,  3363,  3378,  3392,  3395,  3420,  3440,  3461,  3488,  3529,  3531,  3584,  3588,  3591,
         3600,  3602,  3614,  3616,  3628,  3634,  3650,  3657,  3668,  3683,  3685,  3713,  3716,  3720,  3726,  3729,
         3736,  3753,  3778,  3802,  3805,  3819,  3841,  3845,  3851,  3856,  3880,  3922,  3938,  3970,  3993,  4032,
    };

    static const int8_t kvalues_iq3s[8] = { 1, 3, 5, 7, 9, 11, 13, 15 };
    // iq3xxs is similar, but last value would map to 15.5
    static const int8_t kvalues_iq3xxs[8] = { 0x04, 0x0c, 0x14, 0x1c, 0x24, 0x2c, 0x34, 0x3e };

    const int kmap_size = 4096;
    const int nwant = grid_size == 256 ? 2 : 3;
    const uint16_t * kgrid = grid_size == 256 ? kgrid_256 : kgrid_512;
    const int8_t * kvalues = grid_size == 256 ? kvalues_iq3xxs : kvalues_iq3s;
    uint32_t * kgrid_q3xs;
    int      * kmap_q3xs;
    uint16_t * kneighbors_q3xs;

    //printf("================================================================= %s(grid_size = %d)\n", __func__, grid_size);
    kgrid_q3xs = (uint32_t *)malloc(grid_size*sizeof(uint32_t));
    for (int k = 0; k < grid_size; ++k) {
        int8_t * pos = (int8_t *)(kgrid_q3xs + k);
        for (int i = 0; i < 4; ++i) {
            int l = (kgrid[k] >> 3*i) & 0x7;
            pos[i] = kvalues[l];
        }
    }
    iq3_data[gindex].grid = kgrid_q3xs;
    kmap_q3xs = (int *)malloc(kmap_size*sizeof(int));
    iq3_data[gindex].map = kmap_q3xs;
    for (int i = 0; i < kmap_size; ++i) {
        kmap_q3xs[i] = -1;
    }
    for (int i = 0; i < grid_size; ++i) {
        uint16_t index = kgrid[i];
        kmap_q3xs[index] = i;
    }
    int8_t pos[4];
    int * dist2 = (int *)malloc(2*grid_size*sizeof(int));
    int num_neighbors = 0;
    int num_not_in_map = 0;
    for (int i = 0; i < kmap_size; ++i) {
        if (kmap_q3xs[i] >= 0) { continue; }
        ++num_not_in_map;
        for (int k = 0; k < 4; ++k) {
            int l = (i >> 3*k) & 0x7;
            pos[k] = kvalues[l];
        }
        for (int j = 0; j < grid_size; ++j) {
            const int8_t * pg = (const int8_t *)(kgrid_q3xs + j);
            int d2 = 0;
            for (int k = 0; k < 4; ++k) { d2 += (pg[k] - pos[k])*(pg[k] - pos[k]); }
            dist2[2*j+0] = d2;
            dist2[2*j+1] = j;
        }
        qsort(dist2, grid_size, 2*sizeof(int), iq3_compare_func);
        int n = 0; int d2 = dist2[0];
        int nhave = 1;
        for (int j = 0; j < grid_size; ++j) {
            if (dist2[2*j] > d2) {
                if (nhave == nwant) { break; }
                d2 = dist2[2*j];
                ++nhave;
            }
            ++n;
        }
        num_neighbors += n;
    }
    //printf("%s: %d neighbours in total\n", __func__, num_neighbors);
    kneighbors_q3xs = (uint16_t *)malloc((num_neighbors + num_not_in_map)*sizeof(uint16_t));
    iq3_data[gindex].neighbours = kneighbors_q3xs;
    int counter = 0;
    for (int i = 0; i < kmap_size; ++i) {
        if (kmap_q3xs[i] >= 0) { continue; }
        for (int k = 0; k < 4; ++k) {
            int l = (i >> 3*k) & 0x7;
            pos[k] = kvalues[l];
        }
        for (int j = 0; j < grid_size; ++j) {
            const int8_t * pg = (const int8_t *)(kgrid_q3xs + j);
            int d2 = 0;
            for (int k = 0; k < 4; ++k) { d2 += (pg[k] - pos[k])*(pg[k] - pos[k]); }
            dist2[2*j+0] = d2;
            dist2[2*j+1] = j;
        }
        qsort(dist2, grid_size, 2*sizeof(int), iq3_compare_func);
        kmap_q3xs[i] = -(counter + 1);
        int d2 = dist2[0];
        uint16_t * start = &kneighbors_q3xs[counter++];
        int n = 0;
        int nhave = 1;
        for (int j = 0; j < grid_size; ++j) {
            if (dist2[2*j] > d2) {
                if (nhave == nwant) { break; }
                d2 = dist2[2*j];
                ++nhave;
            }
            kneighbors_q3xs[counter++] = dist2[2*j+1];
            ++n;
        }
        *start = n;
    }
    GGML_ASSERT(counter == num_neighbors + num_not_in_map);
    free(dist2);
}

void iq3xs_free_impl(int grid_size) {
    GGML_ASSERT(grid_size == 256 || grid_size == 512);
    const int gindex = iq3_data_index(grid_size);
    if (iq3_data[gindex].grid) {
        free(iq3_data[gindex].grid);       iq3_data[gindex].grid = NULL;
        free(iq3_data[gindex].map);        iq3_data[gindex].map  = NULL;
        free(iq3_data[gindex].neighbours); iq3_data[gindex].neighbours = NULL;
    }
}

// Populate a single neighbour
static void iq3_populate_neighbour(iqx_diff_t *   all_neighbours,
                                   const int8_t * pg,
                                   const float *  xval,
                                   const float *  weight,
                                   const int      grid_index,
                                   const int      g_i) {
    if (all_neighbours[grid_index].index < 0) {
        float sumqx = 0.0f;
        float sumq2 = 0.0f;
        float d2 = 0.0f;
        for (int i = 0; i < 4; ++i) {
            sumqx += weight[i] * xval[i] * pg[i];
            sumq2 += weight[i] * pg[i] * pg[i];
        }
        const float scale = sumq2 > 0.0f ? sumqx / sumq2 : 0.0f;
        for (int i = 0; i < 4; ++i) {
            const float d = pg[i] * scale - xval[i];
            d2 += weight[i] * (d * d);
        }
        all_neighbours[grid_index] = (iqx_diff_t) {
            .scale = scale,
            .d2    = d2,
            .sumqx = sumqx,
            .sumq2 = sumq2,
            .index = grid_index,
            .g_i   = g_i,
        };
    }
}

// Populate all the neighbours of a point
static void iq3_find_neighbours(      iqx_diff_t * GGML_RESTRICT all_neighbours,
                                const uint32_t   * GGML_RESTRICT kgrid,
                                const int        * GGML_RESTRICT kmap,
                                const uint16_t   * GGML_RESTRICT kneighbours,
                                const float      * GGML_RESTRICT xval,
                                const float      * GGML_RESTRICT weight,
                                const int8_t     * GGML_RESTRICT L,
                                const int                        g_i) {
    int u = 0;
    for (int k = 0; k < 4; ++k) {
        u |= ((int) L[k] & 7) << (3 * k);
    }
    int grid_index = kmap[u];
    if (grid_index < 0) {
        const uint16_t * neighbours = kneighbours - (grid_index + 1);
        const uint16_t num_neighbours = neighbours[0];

        for (int i = 1; i <= num_neighbours; ++i) {
            grid_index = neighbours[i];
            // fprintf(stderr, "(%i)", grid_index);
            const int8_t * pg = (const int8_t *) (kgrid + grid_index);
            iq3_populate_neighbour(all_neighbours, pg, xval, weight, grid_index, g_i);
        }
    } else {
        const int8_t * pg = (const int8_t *) (kgrid + grid_index);
        iq3_populate_neighbour(all_neighbours, pg, xval, weight, grid_index, g_i);
    }
}

// sort and prune
// returns the total number of neighbours
static int iqx_sort_neighbours(iqx_diff_t * all_neighbours, int grid_size, int n_idx) {
    int cur = 0;
    int cur_size = 0;
    // gather neighbours into contiguous chunks
    for (int i = 0; i < n_idx; ++i) {
        // gather
        for (int j = 0; j < grid_size; ++j) {
            const int k = i*grid_size + j;
            // ignore zero and negative scales
            if (all_neighbours[k].index > 0) {
                if (k != cur + cur_size) {
                    all_neighbours[cur + cur_size] = all_neighbours[k];
                }
                cur_size += 1;
            }
        }
        // sort by scale
        qsort(all_neighbours + cur, cur_size, sizeof(iqx_diff_t), iqx_diff_scale_compare_asc);

        // prune by distances
        if (cur_size > 1) {
            int new_size = 1;
            for (int j = 1; j < cur_size; j++) {
                const iqx_diff_t * best = all_neighbours + (cur + j - 1);
                const iqx_diff_t * worst = all_neighbours + (cur + j);

                if (worst->d2 < best->d2) {
                    const iqx_diff_t * tmp;
                    tmp = best;
                    best = worst;
                    worst = tmp;
                }

                const float scale_diff = best->scale - worst->scale;
                const float scale_diff2 = scale_diff * scale_diff;

                const float diff2 = best->d2 + best->sumq2*scale_diff2;
                if (diff2 < worst->d2) {
                    // prune worst
                    if (best != all_neighbours + cur + new_size - 1) {
                        all_neighbours[cur + new_size - 1] = *best;
                    }
                } else {
                    // keep both
                    // TODO: reduce copies
                    if (j != new_size) {
                        all_neighbours[cur + new_size - 1] = all_neighbours[cur + j - 1];
                        all_neighbours[cur + new_size] = all_neighbours[cur + j];
                    }
                    new_size += 1;
                }
            }
            cur_size = new_size;
        }
        // FIXME: debug the above

        // calculate scale thresholds
        if (cur_size > 0) {
            // NOTE: the 0th scale is unchanged
            for (int j = 1; j < cur_size; j++) {
                // find the midscale where the distance of both points would be equal
                const float mid_sumqx = all_neighbours[cur + j - 1].sumqx + all_neighbours[cur + j].sumqx;
                const float mid_sumq2 = all_neighbours[cur + j - 1].sumq2 + all_neighbours[cur + j].sumq2;
                const float mid_scale = mid_sumq2 > 0.0f ? mid_sumqx / mid_sumq2 : 0;
                all_neighbours[cur + j].scale = mid_scale;
            }
        }

        cur += cur_size;
        cur_size = 0;
    }

    // sort everything together by scale thresholds
    qsort(all_neighbours, cur, sizeof(iqx_diff_t), iqx_diff_scale_compare_asc);

    return cur;
}

// returns grid_index
// and stores relative sumqx and sumq2 in the respective arguments
// FIXME: need to enumerate the transition scales for each neighbour of L?
//        this needs to depend on the quant_weights,
//        because of the shape of the grid cells.
// FIXME: Need to
static int iq3_find_relative_neighbour(const struct k_sort * GGML_RESTRICT k_sort,
                                       const uint32_t      * GGML_RESTRICT kgrid,
                                       const int           * GGML_RESTRICT kmap,
                                       const uint16_t      * GGML_RESTRICT kneighbours,
                                       const float         * GGML_RESTRICT xval,
                                       const float         * GGML_RESTRICT weight,
                                       const int8_t        * GGML_RESTRICT L,
                                       float               * GGML_RESTRICT sumqx,
                                       float               * GGML_RESTRICT sumq2) {
    int index = 0;
    int8_t p[4];
    for (int i = 0; i < 4; ++i) {
        const int8_t l = L[i] + k_sort->mid_k;
        GGML_ASSERT(l >= 0 && l < 8);
        index |= l << (3 * i);
        p[i] = k_sort->k_values[l];
    }
    int grid_index = kmap[index];

    float sumqx_new = 0.0f;
    float sumq2_new = 0.0f;
    float best_d2 = FLT_MAX;

    if (grid_index < 0) {
        const uint16_t * neighbours = kneighbours - (grid_index + 1);
        const int num_neighbours = neighbours[0];

        // FIXME: what *should* prev_scale be?
        // around what point should the point distances be measured?
        // Why not use the cosine similarity?
        // (because the scale is relevant, due to this being a subset of a block)
        // But why not
        float prev_sumqx = 0.0f;
        float prev_sumq2 = 0.0f;
        float waux[8];
        for (int k = 0; k < 4; ++k) {
            prev_sumqx += weight[k] * (xval[k] * p[k]);
            prev_sumq2 += weight[k] * (p[k] * p[k]);
            waux[k] = sqrtf(weight[k]);
        }
        const float prev_scale = prev_sumq2 > 0.0f ? prev_sumqx / prev_sumq2 : 0.0f;

        for (int i = 1; i <= num_neighbours; ++i) {
            const int8_t * pg = (const int8_t *)(kgrid + neighbours[i]);
            float d2 = 0.0f;
            for (int k = 0; k < 4; ++k) {
                const float diff = prev_scale * pg[k] - xval[k];
                // FIXME: keep track of when the diff would need a bigger scale?
                const float zero_scale = xval[k] / pg[k];
                d2 += waux[k] * diff * diff;
            }

            if (d2 < best_d2) {
                best_d2 = d2;
                grid_index = neighbours[i];
            }
        }

        if (grid_index >= 0) {
            const int8_t * pg = (const int8_t *)(kgrid + grid_index);
            for (int k = 0; k < 4; ++k) {
                const float odd = pg[k] + p[k];
                const float step = pg[k] - p[k];
                sumqx_new += weight[k] * (xval[k] * step);
                sumq2_new += weight[k] * (odd * step);
            }
        }
    }

    *sumqx = sumqx_new;
    *sumq2 = sumq2_new;

    return grid_index;
}

static int iq3_find_best_neighbour(const uint16_t * GGML_RESTRICT neighbours, const uint32_t * GGML_RESTRICT grid,
        const float * GGML_RESTRICT xval, const float * GGML_RESTRICT weight, float scale, int8_t * GGML_RESTRICT L) {
    int num_neighbors = neighbours[0];
    GGML_ASSERT(num_neighbors > 0);
    float best_d2 = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 4; ++i) {
            float q = pg[i];
            float diff = scale*q - xval[i];
            d2 += weight[i]*diff*diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2; grid_index = neighbours[j];
        }
    }
    GGML_ASSERT(grid_index >= 0);
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 4; ++i) L[i] = (pg[i] - 1)/2;
    return grid_index;
}

static float make_iq3_quants(int n, int grid_size, struct k_sort * k_sort, const uint32_t * kgrid, const int * kmap, const uint16_t * kneighbours, const float * xval, const float * weight, int8_t * Laux, iqx_diff_t * all_neighbours, float * sumqx_aux, float * sumq2_aux, int * grid_idx_aux, int * grid_idx) {
    GGML_ASSERT(n % 4 == 0);
    const int n_idx = n / 4;

    for (int i = 0; i < n_idx; ++i) {
        sumqx_aux[i] = 0.0f;
        sumq2_aux[i] = 0.0f;
        grid_idx_aux[i] = 0;
        grid_idx[i] = 0;
    }
    // make all .index fields negative
    memset(all_neighbours, -1, sizeof(iqx_diff_t)*n_idx*grid_size);

    float sumqx = 0.0f;
    float sumq2 = 0.0f;
    float max = 0.0f;
    float w_amax = -1.0f;
    int w_amax_i = -1;
    const float kmin = k_sort->kmin;
    for (int i = 0; i < n; ++i) {
        const float x = xval[i];
        const float w = weight[i];
        const float w_ax = fabsf(x) * w;
        max = MAX(x, max);
        if (w_ax > w_amax) {
            w_amax = w_ax;
            w_amax_i = i;
        }
        sumqx_aux[i / 4] += w * (x * kmin);
        sumqx_aux[i / 4] += w * (kmin * kmin);
        Laux[i] = k_sort->mid_k;
        sumqx += w * (x * kmin);
        sumq2 += w * (kmin * kmin);
    }
    if (max < GROUP_MAX_EPS_IQ3_XXS) {
        return 0.0f;
    }

    k_sort_set_x_L(k_sort, n, w_amax_i, xval, Laux, false);

    float best;
    float best_sumqx;
    float best_sumq2;
    if (sumq2 > 0.0f) {
        best = sumqx * sumqx;
        best_sumqx = sumqx;
        best_sumq2 = sumq2;
    } else {
        best = 0.0f;
        best_sumqx = 0.0f;
        best_sumq2 = 1.0f;
    }
    for (int i = 0; i < k_sort->n; ++i) {
        const int ii = k_sort->ids[i];
        const int k_i = k_sort->k_ids[i];
        const float odd = k_sort->odd[k_i];
        const float step = k_sort->step[k_i];
        const float w = weight[ii];
        const int g_i = ii / 4;
        sumqx += w * (xval[ii] * step);
        sumq2 += w * (odd * step);
        Laux[ii] = k_i;

        iq3_find_neighbours(all_neighbours + grid_size * g_i, kgrid, kmap, kneighbours, xval + 4*g_i, weight + 4*g_i, Laux + 4*g_i, g_i);
    }
    const int num_neighbours = iqx_sort_neighbours(all_neighbours, grid_size, n_idx);
    for (int i = 0; i < num_neighbours; ++i) {
        const int g_i = all_neighbours[i].g_i;
        sumqx_aux[g_i] = all_neighbours[i].sumqx;
        sumq2_aux[g_i] = all_neighbours[i].sumq2;
        grid_idx_aux[g_i] = all_neighbours[i].index;

        float total_sumqx = 0.0f;
        float total_sumq2 = 0.0f;
        for (int k = 0; k < n_idx; ++k) {
            total_sumqx += sumqx_aux[k];
            total_sumq2 += sumq2_aux[k];
        }

        const float current = total_sumqx * total_sumqx;
        if (total_sumq2 > 0.0f && current * best_sumq2 > best * total_sumq2) {
            best = current;
            best_sumqx = total_sumqx;
            best_sumq2 = total_sumq2;
            for (int k = 0; k < n_idx; ++k) {
                grid_idx[k] = grid_idx_aux[k];
            }
        }
    }
    return best_sumq2 > 0.0f ? best_sumqx / best_sumq2 : 0.0f;
}

static float make_iq3_quant(int n, struct k_sort * k_sort, const uint32_t * kgrid, const int * kmap, const uint16_t * kneighbors, const float * xval, const float * weight, int8_t * Laux, float * sumqx_aux, float * sumq2_aux, int * grid_idx_aux, int * grid_idx) {
    GGML_ASSERT(n % 4 == 0);
    const int n_idx = n / 4;

    for (int i = 0; i < n_idx; ++i) {
        sumqx_aux[i] = 0.0f;
        sumq2_aux[i] = 0.0f;
        grid_idx_aux[i] = 0;
        grid_idx[i] = 0;
    }

    float sumqx = 0.0f;
    float sumq2 = 0.0f;
    float max = 0.0f;
    float w_amax = -1.0f;
    int w_amax_i = -1;
    const float kmin = k_sort->kmin;
    for (int i = 0; i < n; ++i) {
        const float x = xval[i];
        const float w = weight[i];
        const float w_ax = fabsf(x) * w;
        max = MAX(x, max);
        if (w_ax > w_amax) {
            w_amax = w_ax;
            w_amax_i = i;
        }
        Laux[i] = k_sort->mid_k;
        sumqx += w * (x * kmin);
        sumq2 += w * (kmin * kmin);
    }
    if (max < GROUP_MAX_EPS_IQ3_XXS) {
        return 0.0f;
    }

    k_sort_set_x_L(k_sort, n, w_amax_i, xval, Laux, false);

    float best;
    float best_sumqx;
    float best_sumq2;
    if (sumq2 > 0.0f) {
        best = sumqx * sumqx;
        best_sumqx = sumqx;
        best_sumq2 = sumq2;
    } else {
        best = 0.0f;
        best_sumqx = 0.0f;
        best_sumq2 = 1.0f;
    }
    for (int i = 0; i < k_sort->n; ++i) {
        const int ii = k_sort->ids[i];
        const int k_i = k_sort->k_ids[i];
        const float odd = k_sort->odd[k_i];
        const float step = k_sort->step[k_i];
        const float w = weight[ii];
        const int g_i = ii / 4;
        sumqx += w * (xval[ii] * step);
        sumq2 += w * (odd * step);
        Laux[ii] = k_i;

        const int grid_index = iq3_find_relative_neighbour(k_sort, kgrid, kmap, kneighbors, xval + 4*g_i, weight + 4*g_i, Laux + 4*g_i, sumqx_aux + g_i, sumq2_aux + g_i);

        if (grid_index == grid_idx_aux[g_i]) { continue; }
        if (grid_index < 0) { break; }

        grid_idx_aux[g_i] = grid_index;

        // avoid some subtraction numerical instabilities by having relative sumqx and sumq2 per grid index
        float sumqx_cur = sumqx;
        float sumq2_cur = sumq2;
        for (int j = 0; j < n_idx; ++j) {
            sumqx_cur += sumqx_aux[j];
            sumq2_cur += sumq2_aux[j];
        }

        const float current = sumqx_cur * sumqx_cur;
        if (sumq2_cur > 0.0f && current * best_sumq2 > best * sumq2_cur) {
            best = current;
            best_sumqx = sumqx_cur;
            best_sumq2 = sumq2_cur;
            for (int j = 0; j < n_idx; ++j) {
                grid_idx[j] = grid_idx_aux[j];
            }
        }
    }

    // FIXME:

    return best_sumq2 > 0.0f ? best_sumqx / best_sumq2 : 0.0f;
}

static void quantize_row_iq3_xxs_impl(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t n,
        const float * GGML_RESTRICT quant_weights) {

    const int gindex = iq3_data_index(256);

    const uint32_t * kgrid_q3xs      = iq3_data[gindex].grid;
    const int      * kmap_q3xs       = iq3_data[gindex].map;
    const uint16_t * kneighbors_q3xs = iq3_data[gindex].neighbours;

    //GGML_ASSERT(quant_weights   && "missing quantization weights");
    GGML_ASSERT(kgrid_q3xs      && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kmap_q3xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbors_q3xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    const int8_t k_values_iq3xxs[8] = { 0x04, 0x0c, 0x14, 0x1c, 0x24, 0x2c, 0x34, 0x3e };
    // actual IQ3_XSS scale kvalues are { 0.25, 0.75, 1.25, ... }, which are odd numbers divided by 4
    const int8_t k_values_iq3xxs_s[16] = { 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31 };

    const int64_t nbl = n/QK_K;

    block_iq3_xxs * y = vy;
    ggml_fp16_t * dh = &y->d;
    uint8_t * qs = y->qs;
    const int block_size = sizeof(block_iq3_xxs);
    const int quant_size = block_size - sizeof(ggml_fp16_t);

    float scales[QK_K/32];
    float sw[QK_K/32];
    int8_t Ls[QK_K/32];
    int8_t Lsaux[QK_K/32];
    float weight[32];
    float xval[32];
    int8_t Laux[32];
    int grid_idx[8];
    int grid_idx_aux[8];
    float sumqx_aux[8];
    float sumq2_aux[8];
    iqx_diff_t all_neighbours[256 * 8];
    uint8_t block_signs[8];
    uint8_t q3[3*(QK_K/8)+QK_K/32];
    uint32_t * scales_and_signs = (uint32_t *)(q3 + QK_K/4);
    struct k_sort k_sort;
    uint8_t buf[K_SORT_BUF_SIZE_NL(32, 8, 7)];
    struct k_sort k_sort_s;
    uint8_t buf_s[K_SORT_BUF_SIZE_NL(QK_K/32, 16, 15)];

    k_sort_init(&k_sort, 32, 8, k_values_iq3xxs, buf);
    k_sort_init(&k_sort_s, QK_K/32, 16, k_values_iq3xxs_s, buf_s);

    for (int ibl = 0; ibl < nbl; ++ibl) {

        dh[0] = GGML_FP32_TO_FP16(0.f);
        memset(q3, 0, 3*QK_K/8+QK_K/32);

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) {
            sumx2 += xbl[i]*xbl[i];
        }
        float sigma2 = 2*sumx2/QK_K;

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const float * xb = xbl + 32*ib;
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*ibl + 32*ib;
                for (int i = 0; i < 32; ++i) {
                    weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
                }
            } else {
                for (int i = 0; i < 32; ++i) {
                    weight[i] = xb[i]*xb[i];
                }
            }
            {
                float sumw = 0.0f;
                for (int i = 0; i < 32; ++i) {
                    sumw += weight[i];
                }
                sw[ib] = sumw;
            }
            for (int k = 0; k < 4; ++k) {
                int nflip = 0;
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) {
                        xval[8*k + i] = xb[8*k + i];
                    } else {
                        xval[8*k + i] = -xb[8*k + i];
                        ++nflip;
                        s |= (1 << i);
                    }
                }
                if (nflip%2) {
                    // uneven parity of flips can't be directly stored
                    // the least significant weight has its sign flipped
                    int imin = 0;
                    float min = weight[8*k + imin] * fabsf(xb[8*k + imin]);
                    for (int i = 1; i < 8; ++i) {
                        float ax = weight[8*k + i] * fabsf(xb[8*k + i]);
                        if (ax < min) {
                            min = ax;
                            imin = i;
                        }
                    }
                    xval[8*k+imin] = -xval[8*k+imin];
                    s ^= (1 << imin);
                }
                block_signs[k] = s & 127;
            }

            const float scale = make_iq3_quants(32, 256, &k_sort, kgrid_q3xs, kmap_q3xs, kneighbors_q3xs, xval, weight, Laux, all_neighbours, sumqx_aux, sumq2_aux, grid_idx_aux, grid_idx);

            for (int k = 0; k < 8; ++k) {
                int grid_index = grid_idx[k];
                q3[8*ib+k] = grid_index;
            }
            scales_and_signs[ib] = block_signs[0] | (block_signs[1] << 7) | (block_signs[2] << 14) | (block_signs[3] << 21);
            GGML_ASSERT(scale >= 0);
            scales[ib] = scale;
        }

        const float d = make_qkxs_nl_quants(QK_K/32, scales, sw, Ls, Lsaux, &k_sort_s, false, true);
        dh[0] = GGML_FP32_TO_FP16(d * 4.0f);
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const int8_t l = Ls[ib];
            scales_and_signs[ib] |= ((uint32_t)l << 28);
        }
        memcpy(qs, q3, quant_size);

        dh += block_size/sizeof(ggml_fp16_t);
        qs += block_size;
    }
}

size_t quantize_iq3_xxs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    int64_t nblock = n_per_row/QK_K;
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq3_xxs_impl(src, qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += nblock*sizeof(block_iq3_xxs);
    }
    return nrow * nblock * sizeof(block_iq3_xxs);
}

void quantize_row_iq3_xxs_ref(const float * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_row_iq3_xxs_impl(x, y, k, NULL);
}

#define IQ3S_BLOCK_SIZE 32
static void quantize_row_iq3_s_impl(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int n,
        const float * GGML_RESTRICT quant_weights) {

    const int gindex = iq3_data_index(512);

    const uint32_t * kgrid_q3xs      = iq3_data[gindex].grid;
    const int      * kmap_q3xs       = iq3_data[gindex].map;
    const uint16_t * kneighbors_q3xs = iq3_data[gindex].neighbours;

    //GGML_ASSERT(quant_weights   && "missing quantization weights");
    GGML_ASSERT(kgrid_q3xs      && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kmap_q3xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbors_q3xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    const int8_t k_values_iq3s[8] = { 0x01, 0x03, 0x05, 0x07, 0x09, 0x0b, 0x0d, 0x0f };
    const int8_t k_values_iq3s_s[16] = { 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31 };

    const int64_t nbl = n / QK_K;

    block_iq3_s * y = vy;

    const int block_size = IQ3S_BLOCK_SIZE;
    const int bs4 = block_size/4;
    const int bs8 = block_size/8;

    float scales[QK_K/IQ3S_BLOCK_SIZE];
    float sw[QK_K/IQ3S_BLOCK_SIZE];
    int8_t Ls[QK_K/IQ3S_BLOCK_SIZE];
    int8_t Lsaux[QK_K/IQ3S_BLOCK_SIZE];
    float weight[IQ3S_BLOCK_SIZE];
    float xval[IQ3S_BLOCK_SIZE];
    int8_t Laux[IQ3S_BLOCK_SIZE];
    int grid_idx[IQ3S_BLOCK_SIZE/4];
    int grid_idx_aux[IQ3S_BLOCK_SIZE/4];
    float sumqx_aux[IQ3S_BLOCK_SIZE/4];
    float sumq2_aux[IQ3S_BLOCK_SIZE/4];
    iqx_diff_t all_neighbours[512 * IQ3S_BLOCK_SIZE/4];
    uint8_t block_signs[IQ3S_BLOCK_SIZE/8];
    struct k_sort k_sort;
    uint8_t buf[K_SORT_BUF_SIZE_NL(IQ3S_BLOCK_SIZE, 8, 7)];
    struct k_sort k_sort_s;
    uint8_t buf_s[K_SORT_BUF_SIZE_NL(QK_K/IQ3S_BLOCK_SIZE, 16, 15)];

    k_sort_init(&k_sort, IQ3S_BLOCK_SIZE, 8, k_values_iq3s, buf);
    k_sort_init(&k_sort_s, QK_K/IQ3S_BLOCK_SIZE, 16, k_values_iq3s_s, buf_s);

    for (int ibl = 0; ibl < nbl; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq3_s));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);

        uint8_t * qs = y[ibl].qs;
        uint8_t * qh = y[ibl].qh;
        uint8_t * signs = y[ibl].signs;

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) {
            sumx2 += xbl[i]*xbl[i];
        }
        float sigma2 = 2*sumx2/QK_K;

        for (int ib = 0; ib < QK_K/block_size; ++ib) {
            const float * xb = xbl + block_size*ib;
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*ibl + block_size*ib;
                for (int i = 0; i < block_size; ++i) {
                    weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
                }
            } else {
                for (int i = 0; i < block_size; ++i) {
                    weight[i] = xb[i]*xb[i];
                }
            }
            {
                float sumw = 0.0f;
                for (int i = 0; i < 32; ++i) {
                    sumw += weight[i];
                }
                sw[ib] = sumw;
            }
            for (int k = 0; k < bs8; ++k) {
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) {
                        xval[8*k + i] = xb[8*k + i];
                    } else {
                        xval[8*k + i] = -xb[8*k + i]; s |= (1 << i);
                    }
                }
                block_signs[k] = s;
            }
            const float scale = make_iq3_quants(block_size, 512, &k_sort, kgrid_q3xs, kmap_q3xs, kneighbors_q3xs, xval, weight, Laux, all_neighbours, sumqx_aux, sumq2_aux, grid_idx_aux, grid_idx);
            for (int k = 0; k < bs4; ++k) {
                int grid_index = grid_idx[k];
                qs[k] = grid_index & 255;
                qh[(ib*bs4+k)/8] |= ((grid_index >> 8) << ((ib*bs4+k)%8));
            }
            qs += bs4;
            for (int k = 0; k < bs8; ++k) {
                signs[k] = block_signs[k];
            }
            signs += bs8;
            GGML_ASSERT(scale >= 0);
            scales[ib] = scale;
        }

        const float d = make_qkxs_nl_quants(QK_K/block_size, scales, sw, Ls, Lsaux, &k_sort_s, false, true);
        y[ibl].d = GGML_FP32_TO_FP16(d);
        for (int ib = 0; ib < QK_K/block_size; ib += 2) {
            const int8_t l1 = Ls[ib+0];
            const int8_t l2 = Ls[ib+1];
            y[ibl].scales[ib/2] = l1 | (l2 << 4);
        }

    }
}

size_t quantize_iq3_s(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    int64_t nblock = n_per_row/QK_K;
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq3_s_impl(src, qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += nblock*sizeof(block_iq3_s);
    }
    return nrow * nblock * sizeof(block_iq3_s);
}

void quantize_row_iq3_s_ref(const float * GGML_RESTRICT x, block_iq3_s * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq3_s(x, y, 1, k, NULL);
}


// =================================== 1.5 bpw ===================================================

static int iq1_find_best_neighbour2(const uint16_t * GGML_RESTRICT neighbours, const uint64_t * GGML_RESTRICT grid,
        const float * GGML_RESTRICT xval, const float * GGML_RESTRICT weight, float scale, const float * GGML_RESTRICT xg, int8_t * GGML_RESTRICT L, int ngrid) {
    int num_neighbors = neighbours[0];
    GGML_ASSERT(num_neighbors > 0);
    float best_score = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 8; ++i) {
            float q = xg[(pg[i] - 1)/2];
            float w = weight[i];
            float diff = scale*q - xval[i];
            d2 += w*diff*diff;
        }
        if (d2 < best_score) {
            best_score = d2;
            grid_index = neighbours[j];
        }
    }
    if (grid_index < 0) {
        for (int i = 0; i < ngrid; ++i) {
            const int8_t * grid_i = (const int8_t *)(grid + i);
            float d2 = 0;
            for (int j = 0; j < 8; ++j) {
                float w = weight[j];
                float q = xg[(grid_i[j] - 1)/2];
                float diff = scale*q - xval[i];
                d2 += w*diff*diff;
            }
            if (d2 < best_score) {
                best_score = d2;
                grid_index = i;
            }
        }
    }
    if (grid_index < 0) {
        printf("Oops, did not find grid point\n");
        printf("Have %d neighbours\n", num_neighbors);
        for (int j = 1; j <= num_neighbors; ++j) {
            const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 8; ++i) {
                float q = xg[(pg[i] - 1)/2];
                float w = weight[i];
                sumqx += w*q*xval[i];
                sumq2 += w*q*q;
            }
            printf("    neighbour %d: sumqx = %g sumq2 = %g\n", j, (double)sumqx, (double)sumq2);
        }
    }
    GGML_ASSERT(grid_index >= 0);
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 8; ++i) L[i] = (pg[i] - 1)/2;
    return grid_index;
}

static int iq1_sort_helper(const void * left, const void * right) {
    const float * l = left;
    const float * r = right;
    return *l < *r ? -1 : *l > *r ? 1 : 0;
}

static float make_iq1_quants(int n, const float * xval, const float * weight, struct k_sort * k_sort, float * sumlx, float * suml2) {
}

#define IQ1S_BLOCK_SIZE 32
#define IQ1M_BLOCK_SIZE 16
static void quantize_row_iq1_s_impl(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t n, const float * GGML_RESTRICT quant_weights,
        float    * scales,
        float    * weight,
        float    * sumx,
        float    * sumw,
        float    * pairs,
        int8_t   * L,
        uint16_t * index,
        int8_t   * shifts) {

    const int gindex = iq2_data_index(GGML_TYPE_IQ1_S);

    const uint64_t * kgrid_q2xs      = iq2_data[gindex].grid;
    const int      * kmap_q2xs       = iq2_data[gindex].map;
    const uint16_t * kneighbors_q2xs = iq2_data[gindex].neighbours;

    GGML_ASSERT(quant_weights   && "missing quantization weights");
    GGML_ASSERT(kgrid_q2xs      && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kmap_q2xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbors_q2xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    block_iq1_s * y = vy;

    const int64_t nbl = n/QK_K;

    const int block_size = IQ1S_BLOCK_SIZE;

    int * idx = (int *)(pairs + 1);

    for (int ibl = 0; ibl < nbl; ++ibl) {

        y[ibl].d = GGML_FP32_TO_FP16(0.f);
        memset(y[ibl].qs, 0, QK_K/8);
        memset(y[ibl].qh, 0, QK_K/16);

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = 2*sumx2/QK_K;

        for (int ib = 0; ib < QK_K/block_size; ++ib) {
            const float * xb = xbl + block_size*ib;
            const float * qw = quant_weights + QK_K*ibl + block_size*ib;
            for (int i = 0; i < block_size; ++i) {
                weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            }

            // FIXME: use something like make_iq2_quants but which checks the two possible offsets

            float max = fabsf(xb[0]);
            for (int i = 1; i < block_size; ++i) max = MAX(max, fabsf(xb[i]));
            if (max < GROUP_MAX_EPS_IQ1_S) {
                scales[ib] = 0;
                memset(L, 1, block_size);
                continue;
            }
            // Here we solve exactly the sum of squared difference (SSD) weighted minimization problem.
            // With just 3 allowed quant values (-1, 0, 1), we can search exhaustively for the two
            // boundaries that split the weights xb[i] into 3 groups. To do so, we sort the weights
            // in ascending order, compute Si = sum[weight[j] xb[j], j = 0...i] and
            // Wi = sum[weight[j], j = 0...i], and use these to quckly get get the optimum scale
            // for each possible and score for each split.
            for (int j = 0; j < block_size; ++j) {
                pairs[2*j] = xb[j];
                idx[2*j] = j;
            }
            qsort(pairs, block_size, 2*sizeof(float), iq1_sort_helper);
            {
                sumx[0] = sumw[0] = 0;
                for (int j = 0; j < block_size; ++j) {
                    int i = idx[2*j];
                    sumx[j+1] = sumx[j] + weight[i]*xb[i];
                    sumw[j+1] = sumw[j] + weight[i];
                }
            }
            float best_score = -FLT_MIN, scale = max;
            int besti1 = -1, besti2 = -1, best_shift = 0;
            // for (int i1 = 0; i1 <= block_size; ++i1) {
            //     for (int i2 = i1; i2 <= block_size; ++i2) {
            //         float sumqx = (sumx[i1] - sumx[0])*x_p[0] + (sumx[i2] - sumx[i1])*x_p[1] + (sumx[block_size] - sumx[i2])*x_p[2];
            //         float sumq2 = (sumw[i1] - sumw[0])*x_p[0]*x_p[0] + (sumw[i2] - sumw[i1])*x_p[1]*x_p[1] + (sumw[block_size] - sumw[i2])*x_p[2]*x_p[2];
            //         if (sumq2 > 0 && sumqx*sumqx > best_score*sumq2) {
            //             scale = sumqx/sumq2; best_score = scale*sumqx;
            //             besti1 = i1; besti2 = i2; best_shift = 1;
            //         }
            //         sumqx = (sumx[i1] - sumx[0])*x_m[0] + (sumx[i2] - sumx[i1])*x_m[1] + (sumx[block_size] - sumx[i2])*x_m[2];
            //         sumq2 = (sumw[i1] - sumw[0])*x_m[0]*x_m[0] + (sumw[i2] - sumw[i1])*x_m[1]*x_m[1] + (sumw[block_size] - sumw[i2])*x_m[2]*x_m[2];
            //         if (sumq2 > 0 && sumqx*sumqx > best_score*sumq2) {
            //             scale = sumqx/sumq2; best_score = scale*sumqx;
            //             besti1 = i1; besti2 = i2; best_shift = -1;
            //         }
            //     }
            // }
            GGML_ASSERT(besti1 >= 0 && besti2 >= 0 && best_shift != 0);
            for (int j =      0; j < besti1; ++j) L[idx[2*j]] = 0;
            for (int j = besti1; j < besti2; ++j) L[idx[2*j]] = 1;
            for (int j = besti2; j < block_size; ++j) L[idx[2*j]] = 2;
            if (scale < 0) {
                for (int j = 0; j < block_size; ++j) L[j] = 2 - L[j];
                scale = -scale; best_shift = -best_shift;
            }
            bool all_on_grid = true;
            // const float * xx = best_shift == 1 ? x_p : x_m;
            // for (int k = 0; k < block_size/8; ++k) {
            //     uint16_t u = 0;
            //     for (int j = 0; j < 8; ++j) u |= (L[8*k+j] << 2*j);
            //     int grid_index = kmap_q2xs[u];
            //     if (grid_index < 0) {
            //         all_on_grid = false;
            //         // const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
            //         const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
            //         grid_index = iq1_find_best_neighbour2(neighbours, kgrid_q2xs, xb + 8*k, weight + 8*k, scale, xx, L + 8*k, NGRID_IQ1S);
            //         GGML_ASSERT(grid_index >= 0);
            //     }
            //     index[k] = grid_index;
            // }
            // if (!all_on_grid) {
            //     float sumqx = 0, sumq2 = 0;
            //     for (int k = 0; k < block_size/8; ++k) {
            //         const int8_t * pg = (const int8_t *)(kgrid_q2xs + index[k]);
            //         for (int j = 0; j < 8; ++j) {
            //             float w = weight[8*k + j];
            //             float q = xx[(pg[j] - 1)/2];
            //             sumqx += w*q*xb[8*k+j];
            //             sumq2 += w*q*q;
            //         }
            //     }
            //     if (sumqx > 0 && sumq2 > 0) scale = sumqx/sumq2;
            // }

            uint16_t h = 0;
            for (int k = 0; k < block_size/8; ++k) {
                y[ibl].qs[(block_size/8)*ib + k] = index[k] & 255;
                h |= (index[k] >> 8) << 3*k;
            }
            y[ibl].qh[ib] = h;
            GGML_ASSERT(scale >= 0);
            scales[ib] = scale;
            shifts[ib] = best_shift;
            max_scale = MAX(max_scale, scale);
        }

        // float d = make_qkxs_nl_quants(QK_K/block_size, scales, sw, Ls, Lsaux, &k_sort, false, true);
        // // y[ibl].d = GGML_FP32_TO_FP16(d*1.125f); // 1.125f is another fudge factor. Don't ask me why it is needed.
        // y[ibl].d = GGML_FP32_TO_FP16(d * 8.0f);
        // for (int ib = 0; ib < QK_K/block_size; ++ib) {
        //     const uint32_t l = Ls[ib];
        //     if (shifts[ib] == -1) l |= 8;
        //     y[ibl].qh[ib] |= (l << 12);
        // }
    }
}

size_t quantize_iq1_s(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    // would use a static assert, but non-integer constant folding is a gnu extension
    GGML_ASSERT(IQ1S_DELTA * 8.0f == 1.0f);
    const int8_t kvalues_iq1s_p[3] = {-7, 1, 9};
    const int8_t kvalues_iq1s_n[3] = {-9, -1, 7};
    const int8_t kvalues_iq1s_s[8] = {1, 3, 5, 7, 9, 11, 13, 15};
    float  scales[QK_K/IQ1S_BLOCK_SIZE];
    float  weight[IQ1S_BLOCK_SIZE];
    int8_t Laux[IQ1S_BLOCK_SIZE];
    float  sumx[IQ1S_BLOCK_SIZE+1];
    float  sumw[IQ1S_BLOCK_SIZE+1];
    float  pairs[2*IQ1S_BLOCK_SIZE];
    uint16_t index[IQ1S_BLOCK_SIZE/8];
    int8_t shifts[QK_K/IQ1S_BLOCK_SIZE];
    struct k_sort k_sort_p;
    struct k_sort k_sort_n;
    struct k_sort k_sort_s;
    uint8_t buf_p[K_SORT_BUF_SIZE_NL(IQ1S_BLOCK_SIZE, 3, 2)];
    uint8_t buf_n[K_SORT_BUF_SIZE_NL(IQ1S_BLOCK_SIZE, 3, 2)];
    uint8_t buf_s[K_SORT_BUF_SIZE_NL(QK_K/IQ1S_BLOCK_SIZE, 8, 7)];
    int64_t nblock = n_per_row/QK_K;
    char * qrow = (char *)dst;
    k_sort_init(&k_sort_p, IQ1S_BLOCK_SIZE, 3, kvalues_iq1s_p, buf_p);
    k_sort_init(&k_sort_n, IQ1S_BLOCK_SIZE, 3, kvalues_iq1s_n, buf_n);
    k_sort_init(&k_sort_s, QK_K/IQ1S_BLOCK_SIZE, 8, kvalues_iq1s_s, buf_s);
    for (int64_t row = 0; row < nrow; ++row) {
        // quantize_row_iq1_s_impl(src, qrow, n_per_row, quant_weights, scales, weight, sumx, sumw, pairs, L, index, shifts);
        src += n_per_row;
        qrow += nblock*sizeof(block_iq1_s);
    }
    return nrow * nblock * sizeof(block_iq1_s);
}

static void quantize_row_iq1_m_impl(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t n, const float * GGML_RESTRICT quant_weights,
        float    * scales,
        float    * weight,
        float    * pairs,
        int8_t   * L,
        uint16_t * index,
        int8_t   * shifts) {

    const int gindex = iq2_data_index(GGML_TYPE_IQ1_M);

    const uint64_t * kgrid_q2xs      = iq2_data[gindex].grid;
    const int      * kmap_q2xs       = iq2_data[gindex].map;
    const uint16_t * kneighbors_q2xs = iq2_data[gindex].neighbours;

    //GGML_ASSERT(quant_weights   && "missing quantization weights");
    GGML_ASSERT(kgrid_q2xs      && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kmap_q2xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbors_q2xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    block_iq1_m * y = vy;

    const int64_t nbl = n/QK_K;

    const int block_size = IQ1M_BLOCK_SIZE;

    const float x_p[3] = {-1 + IQ1M_DELTA,  IQ1M_DELTA, 1 + IQ1M_DELTA};
    const float x_m[3] = {-1 - IQ1M_DELTA, -IQ1M_DELTA, 1 - IQ1M_DELTA};
    const uint8_t masks[4] = {0x00, 0x80, 0x08, 0x88};

    int * idx = (int *)(pairs + 1);

    float sumqx[4], sumq2[4];

    iq1m_scale_t s;
    const float * xx;

    for (int ibl = 0; ibl < nbl; ++ibl) {
        memset(y[ibl].qs, 0, QK_K/8);
        memset(y[ibl].qh, 0, QK_K/16);
        memset(y[ibl].scales, 0, QK_K/32);

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = 2*sumx2/QK_K;

        for (int ib = 0; ib < QK_K/block_size; ++ib) {
            const float * xb = xbl + block_size*ib;
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*ibl + block_size*ib;
                for (int i = 0; i < block_size; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            } else {
                for (int i = 0; i < block_size; ++i) weight[i] = xb[i]*xb[i];
            }
            float max = fabsf(xb[0]);
            for (int i = 1; i < block_size; ++i) max = MAX(max, fabsf(xb[i]));
            if (max < GROUP_MAX_EPS_IQ1_M) {
                scales[ib] = 0;
                memset(L, 1, block_size);
                continue;
            }
            // Here we solve exactly the sum of squared difference (SSD) weighted minimization problem.
            // With just 3 allowed quant values (-1, 0, 1), we can search exhaustively for the two
            // boundaries that split the weights xb[i] into 3 groups. To do so, we sort the weights
            // in ascending order, compute Si = sum[weight[j] xb[j], j = 0...i] and
            // Wi = sum[weight[j], j = 0...i], and use these to quckly get get the optimum scale
            // for each possible and score for each split.
            for (int j = 0; j < block_size; ++j) {
                pairs[2*j] = xb[j];
                idx[2*j] = j;
            }
            qsort(pairs, block_size, 2*sizeof(float), iq1_sort_helper);
            float best_score = -FLT_MIN, scale = max;
            int besti1 = -1, besti2 = -1, best_k = -1;
            // 0: +, +
            // 1: +, -
            // 2: -, +
            // 3: -, -
            for (int i1 = 0; i1 <= block_size; ++i1) {
                for (int i2 = i1; i2 <= block_size; ++i2) {
                    memset(sumqx, 0, 4*sizeof(float));
                    memset(sumq2, 0, 4*sizeof(float));
                    for (int j = 0; j < i1; ++j) {
                        int i = idx[2*j];
                        if (i < block_size/2) {
                            sumqx[0] += weight[i]*x_p[0]*xb[i];
                            sumqx[1] += weight[i]*x_p[0]*xb[i];
                            sumqx[2] += weight[i]*x_m[0]*xb[i];
                            sumqx[3] += weight[i]*x_m[0]*xb[i];
                            sumq2[0] += weight[i]*x_p[0]*x_p[0];
                            sumq2[1] += weight[i]*x_p[0]*x_p[0];
                            sumq2[2] += weight[i]*x_m[0]*x_m[0];
                            sumq2[3] += weight[i]*x_m[0]*x_m[0];
                        } else {
                            sumqx[0] += weight[i]*x_p[0]*xb[i];
                            sumqx[2] += weight[i]*x_p[0]*xb[i];
                            sumqx[1] += weight[i]*x_m[0]*xb[i];
                            sumqx[3] += weight[i]*x_m[0]*xb[i];
                            sumq2[0] += weight[i]*x_p[0]*x_p[0];
                            sumq2[2] += weight[i]*x_p[0]*x_p[0];
                            sumq2[1] += weight[i]*x_m[0]*x_m[0];
                            sumq2[3] += weight[i]*x_m[0]*x_m[0];
                        }
                    }
                    for (int j = i1; j < i2; ++j) {
                        int i = idx[2*j];
                        if (i < block_size/2) {
                            sumqx[0] += weight[i]*x_p[1]*xb[i];
                            sumqx[1] += weight[i]*x_p[1]*xb[i];
                            sumqx[2] += weight[i]*x_m[1]*xb[i];
                            sumqx[3] += weight[i]*x_m[1]*xb[i];
                            sumq2[0] += weight[i]*x_p[1]*x_p[1];
                            sumq2[1] += weight[i]*x_p[1]*x_p[1];
                            sumq2[2] += weight[i]*x_m[1]*x_m[1];
                            sumq2[3] += weight[i]*x_m[1]*x_m[1];
                        } else {
                            sumqx[0] += weight[i]*x_p[1]*xb[i];
                            sumqx[2] += weight[i]*x_p[1]*xb[i];
                            sumqx[1] += weight[i]*x_m[1]*xb[i];
                            sumqx[3] += weight[i]*x_m[1]*xb[i];
                            sumq2[0] += weight[i]*x_p[1]*x_p[1];
                            sumq2[2] += weight[i]*x_p[1]*x_p[1];
                            sumq2[1] += weight[i]*x_m[1]*x_m[1];
                            sumq2[3] += weight[i]*x_m[1]*x_m[1];
                        }
                    }
                    for (int j = i2; j < block_size; ++j) {
                        int i = idx[2*j];
                        if (i < block_size/2) {
                            sumqx[0] += weight[i]*x_p[2]*xb[i];
                            sumqx[1] += weight[i]*x_p[2]*xb[i];
                            sumqx[2] += weight[i]*x_m[2]*xb[i];
                            sumqx[3] += weight[i]*x_m[2]*xb[i];
                            sumq2[0] += weight[i]*x_p[2]*x_p[2];
                            sumq2[1] += weight[i]*x_p[2]*x_p[2];
                            sumq2[2] += weight[i]*x_m[2]*x_m[2];
                            sumq2[3] += weight[i]*x_m[2]*x_m[2];
                        } else {
                            sumqx[0] += weight[i]*x_p[2]*xb[i];
                            sumqx[2] += weight[i]*x_p[2]*xb[i];
                            sumqx[1] += weight[i]*x_m[2]*xb[i];
                            sumqx[3] += weight[i]*x_m[2]*xb[i];
                            sumq2[0] += weight[i]*x_p[2]*x_p[2];
                            sumq2[2] += weight[i]*x_p[2]*x_p[2];
                            sumq2[1] += weight[i]*x_m[2]*x_m[2];
                            sumq2[3] += weight[i]*x_m[2]*x_m[2];
                        }
                    }
                    for (int k = 0; k < 4; ++k) {
                        if (sumq2[k] > 0 && sumqx[k]*sumqx[k] > best_score*sumq2[k]) {
                            scale = sumqx[k]/sumq2[k]; best_score = scale*sumqx[k];
                            besti1 = i1; besti2 = i2; best_k = k;
                        }
                    }
                }
            }
            GGML_ASSERT(besti1 >= 0 && besti2 >= 0 && best_k >= 0);
            for (int j =      0; j < besti1; ++j) L[idx[2*j]] = 0;
            for (int j = besti1; j < besti2; ++j) L[idx[2*j]] = 1;
            for (int j = besti2; j < block_size; ++j) L[idx[2*j]] = 2;
            if (scale < 0) {
                for (int j = 0; j < block_size; ++j) L[j] = 2 - L[j];
                scale = -scale;
                best_k = best_k == 0 ? 3 : best_k == 1 ? 2 : best_k == 2 ? 1 : 0;
            }
            bool all_on_grid = true;
            for (int k = 0; k < block_size/8; ++k) {
                if (k == 0) xx = best_k < 2 ? x_p : x_m;
                else xx = best_k%2 == 0 ? x_p : x_m;
                uint16_t u = 0;
                for (int j = 0; j < 8; ++j) u |= (L[8*k+j] << 2*j);
                int grid_index = kmap_q2xs[u];
                if (grid_index < 0) {
                    all_on_grid = false;
                    // const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                    const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                    grid_index = iq1_find_best_neighbour2(neighbours, kgrid_q2xs, xb + 8*k, weight + 8*k, scale, xx, L + 8*k, NGRID_IQ1S);
                    GGML_ASSERT(grid_index >= 0);
                }
                index[k] = grid_index;
            }
            if (!all_on_grid) {
                float sumqx_f = 0, sumq2_f = 0;
                for (int k = 0; k < block_size/8; ++k) {
                    if (k == 0) xx = best_k < 2 ? x_p : x_m;
                    else xx = best_k%2 == 0 ? x_p : x_m;
                    const int8_t * pg = (const int8_t *)(kgrid_q2xs + index[k]);
                    for (int j = 0; j < 8; ++j) {
                        float w = weight[8*k + j];
                        float q = xx[(pg[j] - 1)/2];
                        sumqx_f += w*q*xb[8*k+j];
                        sumq2_f += w*q*q;
                    }
                }
                if (sumqx_f > 0 && sumq2_f > 0) scale = sumqx_f/sumq2_f;
            }
            y[ibl].qs[2*ib + 0] = index[0] & 255;
            y[ibl].qs[2*ib + 1] = index[1] & 255;
            y[ibl].qh[ib] = (index[0] >> 8) | ((index[1] >> 8) << 4);
            GGML_ASSERT(scale >= 0);
            scales[ib] = scale;
            shifts[ib] = best_k;
            max_scale = MAX(max_scale, scale);
        }

        if (!max_scale) {
            continue;
        }

        // TODO: use make_qkxs_quants here instead
        uint16_t * sc = (uint16_t *)y[ibl].scales;
        float d = max_scale/15;
        float id = 1/d;
        float sumqx_f = 0, sumq2_f = 0;
        for (int ib = 0; ib < QK_K/block_size; ++ib) {
            int l = nearest_int(0.5f*(id*scales[ib+0]-1));
            l = MAX(0, MIN(7, l));
            sc[ib/4] |= (l << 3*(ib%4));
            y[ibl].qh[ib] |= masks[shifts[ib]];
            const float * xb = xbl + block_size*ib;
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*ibl + block_size*ib;
                for (int i = 0; i < block_size; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            } else {
                for (int i = 0; i < block_size; ++i) weight[i] = xb[i]*xb[i];
            }
            for (int k = 0; k < block_size/8; ++k) {
                if (k == 0) xx = shifts[ib] < 2 ? x_p : x_m;
                else xx = shifts[ib]%2 == 0 ? x_p : x_m;
                const int8_t * pg = (const int8_t *)(kgrid_q2xs + y[ibl].qs[2*ib+k] + ((y[ibl].qh[ib] << (8 - 4*k)) & 0x700));
                for (int j = 0; j < 8; ++j) {
                    float w = weight[8*k + j];
                    float q = xx[(pg[j] - 1)/2]*(2*l+1);
                    sumqx_f += w*q*xb[8*k+j];
                    sumq2_f += w*q*q;
                }
            }
        }
        if (sumq2_f > 0) d = sumqx_f/sumq2_f;
        s.f16 = GGML_FP32_TO_FP16(d*1.1125f); // 1.1125f is another fudge factor. Don't ask me why it is needed.
        sc[0] |= ((s.u16 & 0x000f) << 12);
        sc[1] |= ((s.u16 & 0x00f0) <<  8);
        sc[2] |= ((s.u16 & 0x0f00) <<  4);
        sc[3] |= ((s.u16 & 0xf000) <<  0);
    }
}

size_t quantize_iq1_m(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    float  scales[QK_K/IQ1M_BLOCK_SIZE];
    float  weight[IQ1M_BLOCK_SIZE];
    int8_t L[IQ1M_BLOCK_SIZE];
    float  pairs[2*IQ1M_BLOCK_SIZE];
    uint16_t index[IQ1M_BLOCK_SIZE/8];
    int8_t shifts[QK_K/IQ1M_BLOCK_SIZE];
    int64_t nblock = n_per_row/QK_K;
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq1_m_impl(src, qrow, n_per_row, quant_weights, scales, weight, pairs, L, index, shifts);
        src += n_per_row;
        qrow += nblock*sizeof(block_iq1_m);
    }
    return nrow * nblock * sizeof(block_iq1_m);
}

// ============================ 4-bit non-linear quants

static void quantize_row_iq4_nl_impl(const int super_block_size, const int block_size, const float * GGML_RESTRICT x,
        ggml_fp16_t * dh, uint8_t * q4, uint16_t * scales_h, uint8_t * scales_l,
        float * scales, float * weight, uint8_t * L, uint8_t * Laux, struct k_sort * k_sort,
        const float * quant_weights) {

    float sigma2 = 0;
    for (int j = 0; j < super_block_size; ++j) sigma2 += x[j]*x[j];
    sigma2 *= 2.f/super_block_size;

    memset(q4, 0, super_block_size/2);
    dh[0] = GGML_FP32_TO_FP16(0.f);

    float max_scale = 0, amax_scale = 0;
    for (int ib = 0; ib < super_block_size/block_size; ++ib) {
        const float * xb = x + ib*block_size;
        uint8_t * Lb = L + ib*block_size;
        if (quant_weights) {
            const float * qw = quant_weights + ib*block_size;
            for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        } else {
            for (int j = 0; j < block_size; ++j) weight[j] = sqrtf(sigma2 + xb[j]*xb[j]);
        }
        float d = make_qkxs_nl_quants(block_size, xb, weight, (int8_t *) Lb, (int8_t *) Laux, k_sort, true, !quant_weights);
        scales[ib] = d;
        float abs_d = fabsf(d);
        if (abs_d > amax_scale) {
            amax_scale = abs_d; max_scale = d;
        }
    }

    if (super_block_size/block_size > 1) {
        int nb = super_block_size/block_size;
        memset(scales_h, 0, ((nb+7)/8)*sizeof(uint16_t));
        // TODO: use make_qkxs_quants
        float d = -max_scale/32;
        dh[0] = GGML_FP32_TO_FP16(d);
        float id = d ? 1/d : 0.f;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            int l = nearest_int(id*scales[ib]);
            l = MAX(-32, MIN(31, l));
            l += 32;
            uint8_t l_l = l & 0xf;
            uint8_t l_h = l >>  4;
            if (ib%2 == 0) scales_l[ib/2] = l_l;
            else scales_l[ib/2] |= (l_l << 4);
            scales_h[ib/8] |= (l_h << 2*(ib%8));
        }
    } else {
        dh[0] = GGML_FP32_TO_FP16(scales[0]);
    }

    for (int i = 0; i < super_block_size/32; ++i) {
        for (int j = 0; j < 16; ++j) {
            q4[16*i + j] = L[32*i + j] | (L[32*i + 16 + j] << 4);
        }
    }
}

size_t quantize_iq4_nl(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    GGML_ASSERT(n_per_row%QK4_NL == 0);
    int64_t nblock = n_per_row/QK4_NL;
    char * qrow = (char *)dst;
    uint8_t L[QK4_NL];
    uint8_t Laux[QK4_NL];
    struct k_sort k_sort;
    uint8_t buf[K_SORT_BUF_SIZE_NL(QK4_NL, 16, 8)];
    float weight[QK4_NL];
    uint16_t unused_h;
    uint8_t * unused_l = NULL;
    float scale;
    k_sort_init(&k_sort, QK4_NL, 16, kvalues_iq4nl, buf);
    for (int64_t row = 0; row < nrow; ++row) {
        block_iq4_nl * iq4 = (block_iq4_nl *)qrow;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float * qw = quant_weights ? quant_weights + QK4_NL*ibl : NULL;
            quantize_row_iq4_nl_impl(QK4_NL, 32, src + QK4_NL*ibl, &iq4[ibl].d, iq4[ibl].qs, &unused_h, unused_l,
                    &scale, weight, L, Laux, &k_sort, qw);
        }
        src += n_per_row;
        qrow += nblock*sizeof(block_iq4_nl);
    }
    return nrow * nblock * sizeof(block_iq4_nl);
}

//void quantize_row_iq4_nl_ref(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
void quantize_row_iq4_nl_ref(const float * GGML_RESTRICT x, block_iq4_nl * GGML_RESTRICT y, int64_t k) {
    GGML_ASSERT(k%QK4_NL == 0);
    int64_t nblock = k/QK4_NL;
    uint8_t L[QK4_NL];
    uint8_t Laux[QK4_NL];
    struct k_sort k_sort;
    uint8_t buf[K_SORT_BUF_SIZE_NL(QK4_NL, 16, 8)];
    float weight[QK4_NL];
    uint16_t unused_h;
    uint8_t * unused_l = NULL;
    float scale;
    block_iq4_nl * iq4 = y;
    k_sort_init(&k_sort, QK4_NL, 16, kvalues_iq4nl, buf);
    for (int ibl = 0; ibl < nblock; ++ibl) {
        quantize_row_iq4_nl_impl(QK4_NL, 32, x + QK4_NL*ibl, &iq4[ibl].d, iq4[ibl].qs, &unused_h, unused_l,
                &scale, weight, L, Laux, &k_sort, NULL);
    }
}

size_t quantize_iq4_xs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    int64_t nblock = n_per_row/QK_K;
    char * qrow = (char *)dst;
    uint8_t L[QK_K];
    uint8_t Laux[32];
    struct k_sort k_sort;
    uint8_t buf[K_SORT_BUF_SIZE_NL(QK_K, 16, 8)];
    float weight[32];
    float scales[QK_K/32];
    k_sort_init(&k_sort, QK_K, 16, kvalues_iq4nl, buf);
    for (int64_t row = 0; row < nrow; ++row) {
        block_iq4_xs * iq4 = (block_iq4_xs *)qrow;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float * qw = quant_weights ? quant_weights + QK_K*ibl : NULL;
            quantize_row_iq4_nl_impl(QK_K, 32, src + QK_K*ibl, &iq4[ibl].d, iq4[ibl].qs, &iq4[ibl].scales_h, iq4[ibl].scales_l,
                    scales, weight, L, Laux, &k_sort, qw);
        }
        src += n_per_row;
        qrow += nblock*sizeof(block_iq4_xs);
    }
    return nrow * nblock * sizeof(block_iq4_xs);
}

void quantize_row_iq4_xs_ref(const float * GGML_RESTRICT x, block_iq4_xs * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq4_xs(x, y, 1, k, NULL);
}

// =============================== 2.5625 bpw

static void quantize_row_iq2_s_impl(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t n, const float * GGML_RESTRICT quant_weights) {

    const int gindex = iq2_data_index(GGML_TYPE_IQ2_S);

    const uint64_t * kgrid_q2xs       = iq2_data[gindex].grid;
    const int      * kmap_q2xs        = iq2_data[gindex].map;
    const uint16_t * kneighbours_q2xs = iq2_data[gindex].neighbours;

    GGML_ASSERT(kmap_q2xs        && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kgrid_q2xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbours_q2xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    const int8_t k_values_iq2s[3] = { 0x08, 0x19, 0x2b };
    // the quantized scales are in {0.125, 0.375, 0.625, ... }
    // which are the odd numbers divided by 8
    const int8_t k_values_iq2s_s[16] = { 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31 };

    const int64_t nbl = n/QK_K;

    block_iq2_s * y = vy;

    float scales[QK_K/16];
    float sw[QK_K/16];
    int8_t Ls[QK_K/16];
    int8_t Lsaux[QK_K/16];
    float weight[16];
    float xval[16];
    int8_t Laux[16];
    int grid_idx[2];
    int grid_idx_aux[2];
    float sumqx_aux[2];
    float sumq2_aux[2];
    uint8_t block_signs[2];
    struct k_sort k_sort;
    uint8_t buf[K_SORT_BUF_SIZE_NL(16, 3, 2)];
    struct k_sort k_sort_s;
    uint8_t buf_s[K_SORT_BUF_SIZE_NL(QK_K/16, 16, 15)];

    k_sort_init(&k_sort, 16, 3, k_values_iq2s, buf);
    k_sort_init(&k_sort_s, QK_K/16, 16, k_values_iq2s_s, buf_s);

    for (int ibl = 0; ibl < nbl; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq2_s));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) {
            sumx2 += xbl[i]*xbl[i];
        }
        float sigma2 = 2*sumx2/QK_K;

        for (int ib = 0; ib < QK_K/16; ++ib) {
            const float * xb = xbl + 16*ib;
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*ibl + 16*ib;
                for (int i = 0; i < 16; ++i) {
                    weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
                }
            } else {
                for (int i = 0; i < 16; ++i) {
                    weight[i] = 0.25f*sigma2 + xb[i]*xb[i];
                }
            }
            {
                float sumw = 0.0f;
                for (int i = 0; i < 16; ++i) {
                    sumw += weight[i];
                }
                sw[ib] = sumw;
            }
            for (int k = 0; k < 2; ++k) {
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) {
                        xval[8*k + i] = xb[8*k + i];
                    } else {
                        xval[8*k + i] = -xb[8*k + i];
                        s |= (1 << i);
                    }
                }
                block_signs[k] = s;
            }
            const float scale = make_iq2_quants(16, &k_sort, kgrid_q2xs, kmap_q2xs, kneighbours_q2xs, xval, weight, Laux, sumqx_aux, sumq2_aux, grid_idx_aux, grid_idx);

            for (int k = 0; k < 2; ++k) {
                int grid_index = grid_idx[k];
                const int i8 = 2*ib + k;
                y[ibl].qs[i8] = grid_index & 255;
                y[ibl].qh[i8/4] |= ((grid_index >> 8) << 2*(i8%4));
                y[ibl].qs[QK_K/8 + i8] = block_signs[k];
            }

            GGML_ASSERT(scale >= 0);
            scales[ib] = scale;
        }

        const float d = make_qkxs_nl_quants(QK_K/16, scales, sw, Ls, Lsaux, &k_sort_s, false, true);
        y[ibl].d = GGML_FP32_TO_FP16(d * 8.0f);
        for (int ib = 0; ib < QK_K/16; ++ib) {
            const uint8_t l = Ls[ib];
            if (ib % 2 == 0) {
                y[ibl].scales[ib / 2] = l;
            } else {
                y[ibl].scales[ib / 2] |= (l << 4);
            }
        }
    }
}

size_t quantize_iq2_s(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    int64_t nblock = n_per_row/QK_K;
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq2_s_impl(src, qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += nblock*sizeof(block_iq2_s);
    }
    return nrow * nblock * sizeof(block_iq2_s);
}

void quantize_row_iq2_s_ref(const float * GGML_RESTRICT x, block_iq2_s * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq2_s(x, y, 1, k, NULL);
}

// =============================== data validation

static bool validate_float(float f, size_t i) {
    if (isinf(f)) {
        fprintf(stderr, "ggml_validate_row_data: found inf value at block %zu\n", i);
        return false;
    }

    if (isnan(f)) {
        fprintf(stderr, "ggml_validate_row_data: found nan value at block %zu\n", i);
        return false;
    }

    return true;
}

static bool isinf_fp16(ggml_fp16_t f) {
    return (f & 0x7c00) == 0x7c00 && (f & 0x03ff) == 0;
}

static bool isnan_fp16(ggml_fp16_t f) {
    return (f & 0x7c00) == 0x7c00 && (f & 0x03ff) != 0;
}

static bool validate_fp16(ggml_fp16_t f, size_t i) {
    if (isinf_fp16(f)) {
        fprintf(stderr, "ggml_validate_row_data: found inf value at block %zu\n", i);
        return false;
    }

    if (isnan_fp16(f)) {
        fprintf(stderr, "ggml_validate_row_data: found nan value at block %zu\n", i);
        return false;
    }

    return true;
}

static bool validate_e_e8m0(uint8_t e, size_t i) {
    if (e == 0xff) {
        fprintf(stderr, "ggml_validate_row_data: found invalid e value %d at block %zu\n", e, i);
        return false;
    }

    return true;
}

#define VALIDATE_ROW_DATA_D_F16_IMPL(type, data, nb) \
    const type * q = (const type *) (data); \
    for (size_t i = 0; i < (nb); ++i) { \
        if (!validate_fp16(q[i].d, i)) { \
            return false; \
        } \
    }

#define VALIDATE_ROW_DATA_DM_F16_IMPL(type, data, nb, d, m) \
    const type * q = (const type *) (data); \
    for (size_t i = 0; i < (nb); ++i) { \
        if (!validate_fp16(q[i].d, i) || !validate_fp16(q[i].m, i)) { \
            return false; \
        } \
    }

#define VALIDATE_ROW_DATA_E_E8M0_IMPL(type, data, nb) \
    const type * q = (const type *) (data); \
    for (size_t i = 0; i < (nb); ++i) { \
        if (!validate_e_e8m0(q[i].e, i)) { \
            return false; \
        } \
    }

#define VALIDATE_ROW_DATA_DVEC_F16_IMPL(type, data, nb, nr) \
    const type * q = (const type *) (data); \
    for (size_t i = 0; i < (nb); ++i) { \
        for (size_t j = 0; j < (nr); ++j) { \
            if (!validate_fp16(q[i].d[j], i)) { \
                return false; \
            } \
        } \
    }

bool ggml_validate_row_data(enum ggml_type type, const void * data, size_t nbytes) {
    if (type < 0 || type >= GGML_TYPE_COUNT) {
        fprintf(stderr, "%s: invalid type %d\n", __func__, type);
        return false;
    }

    if (nbytes % ggml_type_size(type) != 0) {
        fprintf(stderr, "%s: invalid size %zu for type %s (type size = %zu)\n", __func__, nbytes, ggml_type_name(type), ggml_type_size(type));
        return false;
    }

    const size_t nb = nbytes/ggml_type_size(type);

    switch (type) {
        case GGML_TYPE_BF16:
            {
                int nans = 0;
                int infs = 0;
                const unsigned short * f = (const unsigned short *) data;
                for (size_t i = 0; i < nb; ++i) {
                    nans += (f[i] & 0x7fff) > 0x7f80;
                    infs += (f[i] & 0x7fff) == 0x7f80;
                }
                if (nans) {
                    fprintf(stderr, "%s: found %d NaNs in row of %zu BF16 values\n", __func__, nans, nb);
                    return false;
                }
                if (infs) {
                    fprintf(stderr, "%s: found %d infinities in row of %zu BF16 values\n", __func__, infs, nb);
                    return false;
                }
            } break;
        case GGML_TYPE_F16:
            {
                const ggml_fp16_t * f = (const ggml_fp16_t *) data;
                size_t i = 0;
#if defined(__AVX2__)
                for (; i + 15 < nb; i += 16) {
                    __m256i v = _mm256_loadu_si256((const __m256i *)(f + i));
                    __m256i vexp = _mm256_and_si256(v, _mm256_set1_epi16(0x7c00));
                    __m256i cmp = _mm256_cmpeq_epi16(vexp, _mm256_set1_epi16(0x7c00));
                    int mask = _mm256_movemask_epi8(cmp);
                    if (mask) {
                        for (size_t j = 0; j < 16; ++j) {
                            if (!validate_fp16(f[i + j], i + j)) {
                                return false;
                            }
                        }
                        GGML_UNREACHABLE();
                    }
                }
#elif defined(__ARM_NEON)
                for (; i + 7 < nb; i += 8) {
                    uint16x8_t v = vld1q_u16(f + i);
                    uint16x8_t vexp = vandq_u16(v, vdupq_n_u16(0x7c00));
                    uint16x8_t cmp = vceqq_u16(vexp, vdupq_n_u16(0x7c00));
                    uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(cmp, 4)), 0);
                    if (mask) {
                        for (size_t j = 0; j < 8; ++j) {
                            if (!validate_fp16(f[i + j], i + j)) {
                                return false;
                            }
                        }
                        GGML_UNREACHABLE();
                    }
                }
#endif
                for (; i < nb; ++i) {
                    if (!validate_fp16(f[i], i)) {
                        return false;
                    }
                }
            } break;
        case GGML_TYPE_F32:
            {
                const float * f = (const float *) data;
                size_t i = 0;
#if defined(__AVX2__)
                for (; i + 7 < nb; i += 8) {
                    __m256i v = _mm256_loadu_si256((const __m256i *)(f + i));
                    __m256i vexp = _mm256_and_si256(v, _mm256_set1_epi32(0x7f800000));
                    __m256i cmp = _mm256_cmpeq_epi32(vexp, _mm256_set1_epi32(0x7f800000));
                    int mask = _mm256_movemask_epi8(cmp);
                    if (mask) {
                        for (size_t j = 0; j < 8; ++j) {
                            if (!validate_float(f[i + j], i + j)) {
                                return false;
                            }
                        }
                        GGML_UNREACHABLE();
                    }
                }
#elif defined(__ARM_NEON)
                for (; i + 3 < nb; i += 4) {
                    uint32x4_t v = vld1q_u32((const uint32_t *)f + i);
                    uint32x4_t vexp = vandq_u32(v, vdupq_n_u32(0x7f800000));
                    uint32x4_t cmp = vceqq_u32(vexp, vdupq_n_u32(0x7f800000));
                    uint64_t mask = vget_lane_u64(vreinterpret_u64_u16(vshrn_n_u32(cmp, 8)), 0);
                    if (mask) {
                        for (size_t j = 0; j < 4; ++j) {
                            if (!validate_float(f[i + j], i + j)) {
                                return false;
                            }
                        }
                        GGML_UNREACHABLE();
                    }
                }
#endif
                for (; i < nb; ++i) {
                    if (!validate_float(f[i], i)) {
                        return false;
                    }
                }
            } break;
        case GGML_TYPE_F64:
            {
                const double * f = (const double *) data;
                for (size_t i = 0; i < nb; ++i) {
                    if (!validate_float(f[i], i)) {
                        return false;
                    }
                }
            } break;
        case GGML_TYPE_Q4_0:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_q4_0, data, nb);
            } break;
        case GGML_TYPE_Q4_1:
            {
                VALIDATE_ROW_DATA_DM_F16_IMPL(block_q4_1, data, nb, d, m);
            } break;
        case GGML_TYPE_Q5_0:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_q5_0, data, nb);
            } break;
        case GGML_TYPE_Q5_1:
            {
                VALIDATE_ROW_DATA_DM_F16_IMPL(block_q5_1, data, nb, d, m);
            } break;
        case GGML_TYPE_Q8_0:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_q8_0, data, nb);
            } break;
        case GGML_TYPE_MXFP4:
            {
                VALIDATE_ROW_DATA_E_E8M0_IMPL(block_mxfp4, data, nb);
            } break;
        case GGML_TYPE_Q2_K:
            {
                VALIDATE_ROW_DATA_DM_F16_IMPL(block_q2_K, data, nb, d, dmin);
            } break;
        case GGML_TYPE_Q3_K:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_q3_K, data, nb);
            } break;
        case GGML_TYPE_Q4_K:
            {
                VALIDATE_ROW_DATA_DM_F16_IMPL(block_q4_K, data, nb, d, dmin);
            } break;
        case GGML_TYPE_Q5_K:
            {
                VALIDATE_ROW_DATA_DM_F16_IMPL(block_q5_K, data, nb, d, dmin);
            } break;
        case GGML_TYPE_Q6_K:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_q6_K, data, nb);
            } break;
        case GGML_TYPE_Q8_K:
            {
                const block_q8_K * q = (const block_q8_K *) data;
                for (size_t i = 0; i < nb; ++i) {
                    if (!validate_float(q[i].d, i)) {
                        return false;
                    }
                }
            } break;
        case GGML_TYPE_TQ1_0:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_tq1_0, data, nb);
            } break;
        case GGML_TYPE_TQ2_0:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_tq2_0, data, nb);
            } break;
        case GGML_TYPE_IQ1_S:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_iq1_s, data, nb);
            } break;
        case GGML_TYPE_IQ1_M:
            {
                const block_iq1_m * q = (const block_iq1_m *) data;
                for (size_t i = 0; i < nb; ++i) {
                    iq1m_scale_t scale;
                    const uint16_t * sc = (const uint16_t *)q[i].scales;
                    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
                    if (!validate_fp16(scale.f16, i)) {
                        return false;
                    }
                }
            } break;
        case GGML_TYPE_IQ2_XXS:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_iq2_xxs, data, nb);
            } break;
        case GGML_TYPE_IQ2_XS:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_iq2_xs, data, nb);
            } break;
        case GGML_TYPE_IQ2_S:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_iq2_s, data, nb);
            } break;
        case GGML_TYPE_IQ3_XXS:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_iq3_xxs, data, nb);
            } break;

        case GGML_TYPE_IQ3_S:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_iq3_s, data, nb);
            } break;
        case GGML_TYPE_IQ4_XS:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_iq4_xs, data, nb);
            } break;
        case GGML_TYPE_IQ4_NL:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_iq4_nl, data, nb);
            } break;

        case GGML_TYPE_I8:
        case GGML_TYPE_I16:
        case GGML_TYPE_I32:
        case GGML_TYPE_I64:
            // nothing to validate
            break;
        default:
            {
                fprintf(stderr, "%s: invalid type %d\n", __func__, type);
                return false;
            }
    }

    return true;
}
