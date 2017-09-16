#include <assert.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <math.h>
#include "NN.h"
#include "functions.h"
#include "cublas_v2.h"
#include "kernels.h"

NN::NN(int num_layers_in, std::vector<int> num_nodes_in, int batch_size_in)
{
  cublasStatus_t stat = cublasCreate( &handle );
  // network's configuration
  num_layers = num_layers_in;
  num_nodes = num_nodes_in;
  batch_size = batch_size_in;

  // allocate and initalize the vector d_Ones on device
  float *h_Ones;
  h_Ones = new float[batch_size];
  for(int i = 0; i < batch_size; i++)
  {
    h_Ones[i] = 1;
  }
  cudaMalloc( (void **)&d_Ones, sizeof(float)*batch_size );
  cudaMemcpy(d_Ones, h_Ones, sizeof(float)*batch_size, cudaMemcpyHostToDevice);
  delete[] h_Ones;
  
  // allocate weights and bias on host and device
  h_W = new float* [num_layers-1];
  h_b = new float* [num_layers-1];
  d_W = new float* [num_layers-1];
  d_b = new float* [num_layers-1];
  d_dW = new float* [num_layers-1];
  d_db = new float* [num_layers-1];
  
  for (int l = 0; l < num_layers - 1; l++)
  {
    h_W[l] = new float [num_nodes[l]*num_nodes[l+1]];
    h_b[l] = new float [num_nodes[l+1]];

    cudaMalloc( (void **)&d_W[l],  sizeof(float)*num_nodes[l]*num_nodes[l+1] );
    cudaMalloc( (void **)&d_b[l],  sizeof(float)*num_nodes[l+1] );
    cudaMalloc( (void **)&d_dW[l],  sizeof(float)*num_nodes[l]*num_nodes[l+1] );
    cudaMalloc( (void **)&d_db[l],  sizeof(float)*num_nodes[l+1] );
  }

  
  // initialize the weights and bias on host 
  std::uniform_real_distribution<float> uniform(-1.0,1.0);
  std::string s = "./data/W_";
  for (int l = 0; l < num_layers - 1; l++)
  {
    std::ofstream out_file(s+std::to_string(l)+".txt");
    for(int i = 0; i < num_nodes[l]; i++)
    {
      for(int j = 0; j < num_nodes[l+1]; j++)
      {
	h_W[l][IDX(i,j,num_nodes[l])] = uniform(myGenerator());
	out_file << h_W[l][IDX(i,j,num_nodes[l])] << " ";
      }
      out_file << std::endl;
    }
    out_file.close();
    for(int i = 0; i < num_nodes[l+1]; i++)
    {
      h_b[l][i] = 0;
    }
  }

  // initialize the weights and bias on device
  for (int l = 0; l < num_layers - 1; l++)
  {
    cudaMemcpy( d_W[l], h_W[l], sizeof(float)*num_nodes[l]*num_nodes[l+1], cudaMemcpyHostToDevice);
    cudaMemcpy( d_b[l], h_b[l], sizeof(float)*num_nodes[l+1], cudaMemcpyHostToDevice);
  }

  // allocate activation values for each neuron on device
  d_A = new float* [num_layers];
  for (int l = 0; l < num_layers; l++)
  {
    cudaMalloc( (void **)&d_A[l], sizeof(float)*batch_size*num_nodes[l]);
  }

  // allocate h_p and d_p
  h_p = new float[batch_size*num_nodes[num_layers-1]];
  cudaMalloc( (void **)&d_p, sizeof(float)*batch_size*num_nodes[num_layers-1] );

  // allocate h_predicted_label
  h_predicted_label = new int [batch_size];
    
  // allocate label
  h_label = new int [batch_size];
  cudaMalloc( (void **)&d_label, sizeof(int)*batch_size );
  
  // allocate derivatives
  d_delta = new float* [num_layers-1];
  for (int l = 0; l < num_layers - 1; l++)
  {
    cudaMalloc( (void **)&d_delta[l], sizeof(float)*batch_size*num_nodes[l+1] );
  }    
};

void NN::set_batch(float *batch_data, int *label, int batch_size_in, int dim_in)
{
  assert(batch_size == batch_size_in);
  assert(num_nodes[0] == dim_in);
  cudaMemcpy(d_A[0], batch_data, sizeof(float)*batch_size*num_nodes[0], cudaMemcpyHostToDevice);
  cudaMemcpy(d_label, label, sizeof(int)*batch_size, cudaMemcpyHostToDevice);
  for(int i = 0; i < batch_size; i++)
  {
    h_label[i] = label[i];
  }  
}

void NN::feed_forward()
{
  alpha = 1.0;
  // the following layers 
  for (int l = 1; l < num_layers; l++)
  {
    cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N,
    		 batch_size, num_nodes[l] , num_nodes[l-1],
    		 &alpha,
    		 d_A[l-1], batch_size,
    		 d_W[l-1], num_nodes[l-1],
    		 &beta,
    		 d_A[l], batch_size);
    cublasSger( handle, batch_size, num_nodes[l], &alpha, d_Ones, 1, d_b[l-1], 1, d_A[l], batch_size);
    // the last layer is not a sigmoid layer
    if ( l < num_layers - 1)
    {
      sigmoid<<< batch_size*num_nodes[l]/512 + 1 ,512>>> (d_A[l], batch_size*num_nodes[l]);
    }
  }  
  // softmax for the last layer
  softmax<<< batch_size/512+1,512 >>> (d_A[num_layers-1], d_p, batch_size, num_nodes[num_layers-1]);
  cudaMemcpy(h_p, d_p, sizeof(float)*batch_size*num_nodes[num_layers-1], cudaMemcpyDeviceToHost);  
};

float NN::calc_cost()
{
  float cost = 0;
  for(int i = 0; i < batch_size; i++)
  {
    cost += -log(h_p[IDX(i,h_label[i],batch_size)]);
  }  
  return cost/batch_size;
}

float NN::calc_accuracy()
{
  // cublasIsamax( handle, num_nodes[num_layers-1],
  // 		d_p, batch_size, h_predicted_label);
  // int num_correct_label = 0;
  // for(int i = 0; i < batch_size; i++)
  // {
  //   if (h_predicted_label[i] == h_label[i])
  //   {
  //     num_correct_label++;
  //   }
  //   std::cout <<  num_correct_label << std::endl;
  // }
  // return float(num_correct_label)/batch_size;
  int  num_correct_label = 0;
  for(int i = 0; i < batch_size; i++)
  {
    float max = 0;
    h_predicted_label[i] = 0;
    for(int j = 0; j < num_nodes[num_layers-1]; j++)
    {
      if (h_p[IDX(i,j,batch_size)] > max)
      {
	h_predicted_label[i] = j;
	max = h_p[IDX(i,j,batch_size)];
      }
    }
    if (h_predicted_label[i] == h_label[i])
    {
      num_correct_label++;
    }
  }
  return float(num_correct_label)/batch_size;
}


void NN::back_propogation()
{
  // derivative w.r.t input to the last layer (softmax layer)
  calc_delta_softmax <<< batch_size/512+1,512 >>> (d_label, d_p, batch_size, num_nodes[num_layers-1], d_delta[num_layers-2]);
  
  for(int l = num_layers - 3; l >= 0; l--)
  {
    cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_T,
  		 batch_size, num_nodes[l+1] ,num_nodes[l+2],
  		 &alpha,
  		 d_delta[l+1], batch_size,
  		 d_W[l+1], num_nodes[l+1],
  		 &beta,
  		 d_delta[l], batch_size
  		 );
    calc_delta<<<batch_size*num_nodes[l+1]/512, 512>>>(d_delta[l], d_A[l+1], batch_size, num_nodes[l+1]);
  }

  // derivative w.r.t weights and bias
  alpha = 1.0 / batch_size;
  for(int l = 0; l < num_layers - 1; l++)
  {
    cublasSgemm( handle, CUBLAS_OP_T, CUBLAS_OP_N,
		 num_nodes[l], num_nodes[l+1], batch_size,
		 &alpha,
		 d_A[l], batch_size,
		 d_delta[l], batch_size,
		 &beta,
		 d_dW[l], num_nodes[l]
		 );    
    cublasSgemv( handle, CUBLAS_OP_T,
    		 batch_size, num_nodes[l+1],
    		 &alpha,
    		 d_delta[l], batch_size,
    		 d_Ones, 1,
    		 &beta,
    		 d_db[l],1
		 );
  }
};

float NN::train(int num_steps, float eta)
{
  for (int n = 0; n < num_steps; n++)
  {
    feed_forward();
    back_propogation();
    for (int l = 0; l < num_layers - 1; l++)
    {
      move_one_step<<< num_nodes[l]*num_nodes[l+1]/512 + 1, 512 >>> (d_W[l], d_dW[l], num_nodes[l]*num_nodes[l+1], eta);
      move_one_step<<<num_nodes[l+1]/512 + 1, 512 >>> (d_b[l], d_db[l], num_nodes[l+1], eta);
    }
  }
  return calc_cost();
}

float** NN::get_weights()
{
  for (int l = 0; l < num_layers - 1; l++)
  {
    cudaMemcpy(h_W[l], d_W[l], sizeof(float)*num_nodes[l]*num_nodes[l+1], cudaMemcpyDeviceToHost);
  }
  return h_W;
}

float** NN::get_bias()
{
  for (int l = 0; l < num_layers - 1; l++)
  {
    cudaMemcpy(h_b[l], d_b[l], sizeof(float)*num_nodes[l+1], cudaMemcpyDeviceToHost);
  }
  return h_b;
}

void NN::set_weights(float **weights_in)
{
  for (int l = 0; l < num_layers - 1; l++)
  {
    cudaMemcpy(d_W[l], weights_in[l], sizeof(float)*num_nodes[l]*num_nodes[l+1], cudaMemcpyHostToDevice);
  }
}

void NN::set_bias(float **bias_in)
{
  for (int l = 0; l < num_layers - 1; l++)
  {
    cudaMemcpy(d_b[l], bias_in[l], sizeof(float)*num_nodes[l+1], cudaMemcpyHostToDevice);
  }
}




int NN::get_num_layers()
{
  return num_layers;
}

