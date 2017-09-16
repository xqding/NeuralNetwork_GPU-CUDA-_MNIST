/* @(#)functions.h
 */


#ifndef _FUNCTIONS_H
#define _FUNCTIONS_H 1
#include <random>
#define IDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )
#define CUDA_CALL(F)  if( (F) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__); exit(-1);}
#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__-1); exit(-1);} 
std::default_random_engine& myGenerator();
#endif /* _FUNCTIONS_H */

