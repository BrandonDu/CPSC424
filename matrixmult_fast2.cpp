#include <bits/stdc++.h>
#include <x86intrin.h>
#include <omp.h>
typedef int vector __attribute__((vector_size(64)));

int main() {
  	int n;
	scanf("%d", &n);
    alignas(64) static int a[16000000], b[16000000], c[16000000];


    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            scanf("%d", &a[j * n + i]);
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            scanf("%d", &b[j * n + i]);
        }
    }


    int nx = (n + 5) / 6 * 6;
    int ny = (n + 15) / 16 * 16;

    const int s3 = 48;
    const int s2 = 2048;
    const int s1 = 512;

    vector *vec_b = (vector *)b;
    vector *vec_c = (vector *)c;

    #pragma omp parallel for collapse(2)
    for (int i3 = 0; i3 < ny; i3 += s3)
        for (int i2 = 0; i2 < nx; i2 += s2)
            for (int i1 = 0; i1 < ny; i1 += s1)
                for (int x = i2; x < i2 + s2; x += 6)
                    for (int y = i3; y < i3 + s3; y += 16) {
                        vector t00, t01,
                               t10, t11,
                               t20, t21,
                               t30, t31,
                               t40, t41,
                               t50, t51;

                        t00 = vec_c[((x + 0) * nx + y) / 8 + 0];
                        t01 = vec_c[((x + 0) * nx + y) / 8 + 1];

                        t10 = vec_c[((x + 1) * nx + y) / 8 + 0];
                        t11 = vec_c[((x + 1) * nx + y) / 8 + 1];

                        t20 = vec_c[((x + 2) * nx + y) / 8 + 0];
                        t21 = vec_c[((x + 2) * nx + y) / 8 + 1];

                        t30 = vec_c[((x + 3) * nx + y) / 8 + 0];
                        t31 = vec_c[((x + 3) * nx + y) / 8 + 1];

                        t40 = vec_c[((x + 4) * nx + y) / 8 + 0];
                        t41 = vec_c[((x + 4) * nx + y) / 8 + 1];

                        t50 = vec_c[((x + 5) * nx + y) / 8 + 0];
                        t51 = vec_c[((x + 5) * nx + y) / 8 + 1];

                        for (int k = i1; k < i1 + s1; k += 2) {
                            vector a0_0 = vector{} + a[(x + 0) * nx + k];
                            vector a0_1 = vector{} + a[(x + 0) * nx + k + 1];

                            t00 += a0_0 * vec_b[(k * nx + y) / 8];
                            t00 += a0_1 * vec_b[((k + 1) * nx + y) / 8];
                            t01 += a0_0 * vec_b[(k * nx + y) / 8 + 1];
                            t01 += a0_1 * vec_b[((k + 1) * nx + y) / 8 + 1];

                            vector a1_0 = vector{} + a[(x + 1) * nx + k];
                            vector a1_1 = vector{} + a[(x + 1) * nx + k + 1];
                            t10 += a1_0 * vec_b[(k * nx + y) / 8];
                            t10 += a1_1 * vec_b[((k + 1) * nx + y) / 8];
                            t11 += a1_0 * vec_b[(k * nx + y) / 8 + 1];
                            t11 += a1_1 * vec_b[((k + 1) * nx + y) / 8 + 1];

                            vector a2_0 = vector{} + a[(x + 2) * nx + k];
                            vector a2_1 = vector{} + a[(x + 2) * nx + k + 1];
                            t20 += a2_0 * vec_b[(k * nx + y) / 8];
                            t20 += a2_1 * vec_b[((k + 1) * nx + y) / 8];
                            t21 += a2_0 * vec_b[(k * nx + y) / 8 + 1];
                            t21 += a2_1 * vec_b[((k + 1) * nx + y) / 8 + 1];

                            vector a3_0 = vector{} + a[(x + 3) * nx + k];
                            vector a3_1 = vector{} + a[(x + 3) * nx + k + 1];
                            t30 += a3_0 * vec_b[(k * nx + y) / 8];
                            t30 += a3_1 * vec_b[((k + 1) * nx + y) / 8];
                            t31 += a3_0 * vec_b[(k * nx + y) / 8 + 1];
                            t31 += a3_1 * vec_b[((k + 1) * nx + y) / 8 + 1];

                            vector a4_0 = vector{} + a[(x + 4) * nx + k];
                            vector a4_1 = vector{} + a[(x + 4) * nx + k + 1];
                            t40 += a4_0 * vec_b[(k * nx + y) / 8];
                            t40 += a4_1 * vec_b[((k + 1) * nx + y) / 8];
                            t41 += a4_0 * vec_b[(k * nx + y) / 8 + 1];
                            t41 += a4_1 * vec_b[((k + 1) * nx + y) / 8 + 1];

                            vector a5_0 = vector{} + a[(x + 5) * nx + k];
                            vector a5_1 = vector{} + a[(x + 5) * nx + k + 1];
                            t50 += a5_0 * vec_b[(k * nx + y) / 8];
                            t50 += a5_1 * vec_b[((k + 1) * nx + y) / 8];
                            t51 += a5_0 * vec_b[(k * nx + y) / 8 + 1];
                            t51 += a5_1 * vec_b[((k + 1) * nx + y) / 8 + 1];
                        }

                        vec_c[((x + 0) * nx + y) / 8 + 0] = t00;
                        vec_c[((x + 0) * nx + y) / 8 + 1] = t01;

                        vec_c[((x + 1) * nx + y) / 8 + 0] = t10;
                        vec_c[((x + 1) * nx + y) / 8 + 1] = t11;

                        vec_c[((x + 2) * nx + y) / 8 + 0] = t20;
                        vec_c[((x + 2) * nx + y) / 8 + 1] = t21;

                        vec_c[((x + 3) * nx + y) / 8 + 0] = t30;
                        vec_c[((x + 3) * nx + y) / 8 + 1] = t31;

                        vec_c[((x + 4) * nx + y) / 8 + 0] = t40;
                        vec_c[((x + 4) * nx + y) / 8 + 1] = t41;

                        vec_c[((x + 5) * nx + y) / 8 + 0] = t50;
                        vec_c[((x + 5) * nx + y) / 8 + 1] = t51;
                    }


    std::ios::sync_with_stdio(false);
    std::ostringstream buffer;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            buffer << c[i * nx + j] << " ";
        }
        buffer << '\n';
    }

    std::cout << buffer.str();
    return 0;
}
