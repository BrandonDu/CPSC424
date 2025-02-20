

#include <iostream>
#include <vector>

#include <omp.h>

int main() {
    int n;
    std::cin >> n;

    std::vector<std::vector<int>> A(n, std::vector<int>(n));
    std::vector<std::vector<int>> B(n, std::vector<int>(n));
    std::vector<std::vector<int>> C(n, std::vector<int>(n, 0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cin >> A[i][j];
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cin >> B[i][j];
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
      for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
          C[i][j] += A[i][k] * B[k][j];
        }
      }
    }


    std::cout << "The resulting matrix C = A x B is:\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << C[i][j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
