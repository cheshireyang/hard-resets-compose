/* Single-threaded CPU reference kernels for the hard-reset LIF (float32, layout [T, N] row-major).
   Forward: v = beta*u + I_t, s = [v >= theta], u <- s ? u_r : v.
   Backward (detached reset, sigmoid surrogate with slope alpha, dL/du_t = 0):
     lam_t = gs_t * sg(v_t - theta) + beta * (1 - s_t) * lam_{t+1}.
   Build: cc -O3 -march=native -ffast-math -shared -fPIC -o liblifcpu.so lif_cpu.c -lm */
#include <stdint.h>
#include <math.h>

void lif_fwd(const float *I, float *U, float *S, const float *u0, int64_t T, int64_t N,
             float beta, float theta, float ur) {
    float u[N];
    for (int64_t n = 0; n < N; n++) u[n] = u0[n];
    for (int64_t t = 0; t < T; t++) {
        const float *x = I + t * N; float *uo = U + t * N; float *so = S + t * N;
        for (int64_t n = 0; n < N; n++) {
            float v = beta * u[n] + x[n];
            int s = v >= theta;
            u[n] = s ? ur : v;
            uo[n] = u[n]; so[n] = (float)s;
        }
    }
}

void lif_bwd(const float *I, const float *U, const float *S, const float *GS, const float *u0, float *Lam,
             int64_t T, int64_t N, float beta, float theta, float alpha) {
    float lam[N];
    for (int64_t n = 0; n < N; n++) lam[n] = 0.0f;
    for (int64_t t = T - 1; t >= 0; t--) {
        const float *x = I + t * N, *s = S + t * N, *gs = GS + t * N;
        const float *up = t > 0 ? U + (t - 1) * N : u0;
        float *lo = Lam + t * N;
        for (int64_t n = 0; n < N; n++) {
            float v = beta * up[n] + x[n];
            float sig = 1.0f / (1.0f + expf(-alpha * (v - theta)));
            lam[n] = gs[n] * alpha * sig * (1.0f - sig) + beta * (1.0f - s[n]) * lam[n];
            lo[n] = lam[n];
        }
    }
}
