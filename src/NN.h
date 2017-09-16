/* @(#)NN.h
 */

#ifndef _NN_H
#define _NN_H 1
#include <vector>
#include "cublas_v2.h"
class NN
{
  int num_layers; // num of layers
  std::vector<int> num_nodes; // num of nodes in each layers
  int batch_size;
  int *d_label; // label on device
  int *h_label; // label on host
  float **h_W, **h_b; // weights and bias on host
  float **d_W, **d_b; // weights and bias on device
  float **d_A; // activation value for each neuron
  float *h_p,*d_p; // softmax probability for the last layer
  int *h_predicted_label; // predicted label for current batch data
  float *d_Ones; // vector with elements of 1's

  float **d_dW; // derivative with respect to weights W
  float **d_db; // derivative with respect to bias b
  float **d_delta; // derivative with respect to input to each neuron
  cublasHandle_t handle;
  float alpha = 1.0;
  float beta = 0.0;

 public:
  NN(int num_layers_in, std::vector<int> num_nodes_in, int batch_size_in);
  int get_num_layers();
  void set_batch(float *batch_data, int *label, int batch_size_in, int dim_in);
  float calc_cost();
  float calc_accuracy();
  void feed_forward();
  void back_propogation();
  float train(int num_steps, float eta);
  float** get_weights();
  float** get_bias();
  void set_weights(float **weights_in);
  void set_bias(float **bias_in);
};

#endif /* _NN_H */

