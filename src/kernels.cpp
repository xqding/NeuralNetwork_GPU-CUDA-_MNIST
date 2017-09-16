#include "kernels.h"
#include "functions.h"

__global__ void sigmoid(float *A, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    A[idx] = 1 / (1 + expf(-A[idx]));
  }
};

__global__ void expM(float *A, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    A[idx] = expf(A[idx]);
  }
};

__global__ void softmax(float *A, float* p, int m, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m)
  {
    float max = A[IDX(idx,0,m)];
    for ( int j = 1; j < n; j++)
    {
      if (A[IDX(idx,j,m)] > max)
      {
	max = A[IDX(idx,j,m)];
      }
    }
    float sum = 0;
    for(int j = 0; j < n; j++)
    {
      p[IDX(idx,j,m)] = expf(A[IDX(idx,j,m)] - max);
      sum = sum + p[IDX(idx,j,m)];
    }

    for(int j = 0; j < n; j++)
    {
      p[IDX(idx,j,m)] = p[IDX(idx,j,m)] / sum;
    }
  }
}

__global__ void calc_delta_softmax(int *label, float* p, int m, int n, float* delta)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m)
  {
    for (int j = 0; j < n; j++)
    {
      if (label[idx] == j )
	delta[IDX(idx,j,m)] = 1 - p[IDX(idx,j,m)];
      else
	delta[IDX(idx,j,m)] = - p[IDX(idx,j,m)];
    }
  }
}

__global__ void calc_delta(float *delta, float* A, int m, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m*n)
  {
    delta[idx] = delta[idx] * A[idx] * (1-A[idx]);
  }
}

__global__ void move_one_step(float *w, float *dw, int n, float eta)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    w[idx] = w[idx] + eta * dw[idx];
  }
}
