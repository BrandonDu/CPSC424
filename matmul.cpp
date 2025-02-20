
#include <bits/stdc++.h>
#include <x86intrin.h>
#include <omp.h>

typedef float vector __attribute__ (( vector_size(64) ));

void kernel(float *a, vector *b, vector *c, int x, int y, int l, int r, int n) {
    vector t00, t01,
           t10, t11,
           t20, t21,
           t30, t31,
           t40, t41,
           t50, t51;

    t00 = c[((x + 0) * n + y) / 8 + 0];
    t01 = c[((x + 0) * n + y) / 8 + 1];

    t10 = c[((x + 1) * n + y) / 8 + 0];
    t11 = c[((x + 1) * n + y) / 8 + 1];

    t20 = c[((x + 2) * n + y) / 8 + 0];
    t21 = c[((x + 2) * n + y) / 8 + 1];

    t30 = c[((x + 3) * n + y) / 8 + 0];
    t31 = c[((x + 3) * n + y) / 8 + 1];

    t40 = c[((x + 4) * n + y) / 8 + 0];
    t41 = c[((x + 4) * n + y) / 8 + 1];

    t50 = c[((x + 5) * n + y) / 8 + 0];
    t51 = c[((x + 5) * n + y) / 8 + 1];

    for (int k = l; k < r; k++) {
        vector a0 = vector{} + a[(x + 0) * n + k];
        t00 += a0 * b[(k * n + y) / 8];
        t01 += a0 * b[(k * n + y) / 8 + 1];

        vector a1 = vector{} + a[(x + 1) * n + k];
        t10 += a1 * b[(k * n + y) / 8];
        t11 += a1 * b[(k * n + y) / 8 + 1];

        vector a2 = vector{} + a[(x + 2) * n + k];
        t20 += a2 * b[(k * n + y) / 8];
        t21 += a2 * b[(k * n + y) / 8 + 1];

        vector a3 = vector{} + a[(x + 3) * n + k];
        t30 += a3 * b[(k * n + y) / 8];
        t31 += a3 * b[(k * n + y) / 8 + 1];

        vector a4 = vector{} + a[(x + 4) * n + k];
        t40 += a4 * b[(k * n + y) / 8];
        t41 += a4 * b[(k * n + y) / 8 + 1];

        vector a5 = vector{} + a[(x + 5) * n + k];
        t50 += a5 * b[(k * n + y) / 8];
        t51 += a5 * b[(k * n + y) / 8 + 1];
    }

    c[((x + 0) * n + y) / 8 + 0] = t00;
    c[((x + 0) * n + y) / 8 + 1] = t01;

    c[((x + 1) * n + y) / 8 + 0] = t10;
    c[((x + 1) * n + y) / 8 + 1] = t11;

    c[((x + 2) * n + y) / 8 + 0] = t20;
    c[((x + 2) * n + y) / 8 + 1] = t21;

    c[((x + 3) * n + y) / 8 + 0] = t30;
    c[((x + 3) * n + y) / 8 + 1] = t31;

    c[((x + 4) * n + y) / 8 + 0] = t40;
    c[((x + 4) * n + y) / 8 + 1] = t41;

    c[((x + 5) * n + y) / 8 + 0] = t50;
    c[((x + 5) * n + y) / 8 + 1] = t51;
}

void matmul(const float *_a, const float *_b, float *_c, int n) {

}

int main() {
    int n;
    scanf("%d", &n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            scanf("%f", &a[j * n + i]);
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            scanf("%f", &b[j * n + i]);
        }
    }

    int nx = (n + 5) / 6 * 6;
    int ny = (n + 15) / 16 * 16;

    const int MAXN = 4000 * 4000;
    alignas(64) static float a[MAXN], b[MAXN], c[MAXN];

    const int s3 = 48;
    const int s2 = 2048;
    const int s1 = 512;

    #pragma omp parallel for collapse(2)
    for (int i3 = 0; i3 < ny; i3 += s3)
        for (int i2 = 0; i2 < nx; i2 += s2)
            for (int i1 = 0; i1 < ny; i1 += s1)
                for (int x = i2; x < i2 + s2; x += 6)
                    for (int y = i3; y < i3 + s3; y += 16)
                        kernel(a, (vector*) b, (vector*) c, x, y, i1, i1 + s1, ny);


    std::ios::sync_with_stdio(false);
    std::ostringstream buffer;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            buffer << c[i][j] << " ";
        }
        buffer << '\n';
    }

    std::cout << buffer.str();
    return 0;
}