// MP 2: Due Sunday, Dec 16, 2012 at 11:59 p.m. PST
#include    <wb.h>

#define TILE_WIDTH 16

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

// Compute C = A * B
__global__ void matrixMultiply(float * A, float * B, float * C,
                   int numARows, int numAColumns,
                   int numBRows, int numBColumns,
                   int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < numCRows) && (col < numCColumns)) {
        float Cvalue = 0;
        for (int k = 0; k < numAColumns; ++k) {
            Cvalue += A[row * numAColumns + k] * B[k * numBColumns + col];
        }
        C[row * numCColumns + col] = Cvalue;
    }
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;

    // If A is an n x m matrix and B is an m x p matrix, the result AB of
    // their multiplication is an n x p matrix
    int numARows; // number of rows in the matrix A (n)
    int numAColumns; // number of columns in the matrix A (m)
    int numBRows; // number of rows in the matrix B - (m)
    int numBColumns; // number of columns in the matrix B - (p)
    int numCRows; // number of rows in the matrix C (n)
    int numCColumns; // number of columns in the matrix C (p)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    //@@ Allocate the hostC matrix
    int sizeA = numARows * numAColumns * sizeof(float);
    int sizeB = numBRows * numBColumns * sizeof(float);
    int sizeC = numCRows * numCColumns * sizeof(float);
    hostC = (float *) malloc(sizeC);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    cudaMalloc((void **) &deviceA, sizeA);
    cudaMalloc((void **) &deviceB, sizeB);
    cudaMalloc((void **) &deviceC, sizeC);

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);

    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    dim3 DimGrid(numCRows/TILE_WIDTH, numCColumns/TILE_WIDTH, 1);
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    matrixMultiply<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC,
                   numARows, numAColumns,
                   numBRows, numBColumns,
                   numCRows, numCColumns);
    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    wbCheck(cudaGetLastError());

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}
