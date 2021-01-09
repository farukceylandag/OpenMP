#include <omp.h>
#include <ctime>
#include <iostream>

using namespace std;

#define N 1000
#define thread_num omp_get_max_threads() //Thread Num

template <class M>
class Matrix {

public:

    Matrix();
    void Matmul();
    void Parallel_Matmul();

private:

    static int i, j, k;
    static M A[N][N];
    static M B[N][N];
    static M C[N][N];
    M value = 1.0;

};

template <class M> int Matrix<M>::i;
template <class M> int Matrix<M>::j;
template <class M> int Matrix<M>::k;

template <class M> M Matrix<M>::A[N][N];
template <class M> M Matrix<M>::B[N][N];
template <class M> M Matrix<M>::C[N][N];

template <class M>
Matrix<M>::Matrix() {

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++)
        {
            A[i][j] = value;
            B[i][j] = value;
            C[i][j] = 0;
        }
    }
}

template <class M>
void Matrix<M>::Matmul() {

    clock_t start = clock(), end;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    end = clock();

    if (sizeof(M) == sizeof(float))
        cout << "Float Serial Runnig Time = " << float(end - start) / CLOCKS_PER_SEC << " sec" << endl;
    else if (sizeof(M) == sizeof(double))
        cout << "Double Serial Running Time = " << float(end - start) / CLOCKS_PER_SEC << " sec" << endl;
}

template <class M>
void Matrix<M>::Parallel_Matmul() {

    clock_t start = clock(), end;
#pragma omp parallel for private(i,j,k) shared(A,B,C)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    end = clock();

    if (sizeof(M) == sizeof(float))
        cout << "Float Parallel Running Time = " << float(end - start) / (CLOCKS_PER_SEC) << " sec" << endl;
    else if (sizeof(M) == sizeof(double))
        cout << "Double Parallel Runnig Time = " << float(end - start) / (CLOCKS_PER_SEC) << " sec" << endl;
}

int main() {

    omp_set_num_threads(thread_num);
    cout << "\nThread Num = " << omp_get_max_threads() << "\n" << endl;
    cout << "For " << N << "x" << N << " Matrix;\n---------------------" << endl;

    Matrix<float>* fm = new Matrix<float>(); //Float Matrix
    fm->Matmul();
    fm->Parallel_Matmul();

    cout << endl;

    Matrix<double>* dm = new Matrix<double>(); //Double Matrix
    dm->Matmul();
    dm->Parallel_Matmul();

    return 0;
}
