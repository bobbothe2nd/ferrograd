#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int BLOCK_THREADS_X = 16;
constexpr int BLOCK_THREADS_Y = 16;

constexpr int BLOCK_THREADS = BLOCK_THREADS_X * BLOCK_THREADS_Y;

constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / 32;

constexpr int BLOCK_M = 32;
constexpr int BLOCK_N = 64;

__global__
void matmul_wmma(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    const int thread_x = threadIdx.x;
    const int thread_y = threadIdx.y;

    const int thread_linear_id =
        thread_y * BLOCK_THREADS_X + thread_x;

    const int warp_id =
        thread_linear_id / 32;

    const int lane_id =
        thread_linear_id % 32;

    (void)lane_id;

    const int block_row =
        blockIdx.y * BLOCK_M;

    const int block_col =
        blockIdx.x * BLOCK_N;

    const int warp_tile_row =
        warp_id / 4;

    const int warp_tile_col =
        warp_id % 4;

    const int c_tile_row =
        block_row + warp_tile_row * WMMA_M;

    const int c_tile_col =
        block_col + warp_tile_col * WMMA_N;

    __shared__ half shared_A[32 * 16];

    __shared__ half shared_B[16 * 64];

    wmma::fragment<
        wmma::matrix_a,
        16,
        16,
        16,
        half,
        wmma::row_major
    > a_fragment;

    wmma::fragment<
        wmma::matrix_b,
        16,
        16,
        16,
        half,
        wmma::row_major
    > b_fragment;

    wmma::fragment<
        wmma::accumulator,
        16,
        16,
        16,
        float
    > accumulator;


    wmma::fill_fragment(
        accumulator,
        0.0f
    );

    for (int k_base = 0;
         k_base < K;
         k_base += 16) {

        for (int index = thread_linear_id;
             index < 32 * 16;
             index += BLOCK_THREADS) {

            const int local_row =
                index / 16;

            const int local_col =
                index % 16;

            const int global_row =
                block_row + local_row;

            const int global_col =
                k_base + local_col;

            shared_A[index] =
                A[
                    global_row * K
                    + global_col
                ];
        }

        for (int index = thread_linear_id;
             index < 16 * 64;
             index += BLOCK_THREADS) {

            const int local_row =
                index / 64;

            const int local_col =
                index % 64;

            const int global_row =
                k_base + local_row;

            const int global_col =
                block_col + local_col;

            shared_B[index] =
                B[
                    global_row * N
                    + global_col
                ];
        }

        __syncthreads();

        const int shared_A_offset =
            warp_tile_row * 16 * 16;

        const int shared_B_offset =
            warp_tile_col * 16;

        wmma::load_matrix_sync(
            a_fragment,
            &shared_A[shared_A_offset],
            16
        );

        wmma::load_matrix_sync(
            b_fragment,
            &shared_B[shared_B_offset],
            64
        );

        wmma::mma_sync(
            accumulator,
            a_fragment,
            b_fragment,
            accumulator
        );

        __syncthreads();
    }

    float* C_tile =
        C
        + c_tile_row * N
        + c_tile_col;

    wmma::store_matrix_sync(
        C_tile,
        accumulator,
        N,
        wmma::mem_row_major
    );
}
