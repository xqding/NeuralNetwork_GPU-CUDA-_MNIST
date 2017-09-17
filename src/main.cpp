#include <iostream>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <stdlib.h>
#include <random>
#include "cublas_v2.h"
#include "NN.h"
#include <vector>
#include "functions.h"

int main(int argc, char** argv){
  std::vector<int> num_nodes;
  int dim = 784;
  
  num_nodes.push_back(dim); // input layer

  ///////////////////////////////////////////////////////////////////////////////////
  //  specify the num of hidden layers and the num of neurons in each hidden layer // 
  ///////////////////////////////////////////////////////////////////////////////////
  // 
  num_nodes.push_back(30); 
  // If you want to add another hidder layer with 30 neurons, you can
  // uncomment the following line. You can add as many hidden layers
  // as you want.
  
  // num_nodes.push_back(30); // add one hidder layer with 30 neurons
  
  num_nodes.push_back(10); // the softmax layer

  /////////////////////////////
  //  Optimization parameter // 
  /////////////////////////////
  int num_steps = 1000; // num of gradien descent steps
  float epsilon = 2.0; // step size


  //// print configuration of the network
  std::cout << "////////////////////////////////////////////////////////////" << std::endl;
  std::cout << "Configuration of the nerual network:" << std::endl;
  std::cout << "Num of layers: " << num_nodes.size() << std::endl;
  std::cout << "Num of neurons in each layer: " << std::endl ;
  std::cout << "  size of input layer: " << 784 << std::endl;
  for (int i = 1; i < num_nodes.size() - 1; i++)
  {
    std::cout << "  size of hidden layer " << i << ": " << num_nodes[i] << std::endl;
  }
  std::cout << "  size of softmax layer: " << 10 << std::endl;;

  std::cout << "Parameter for of optimization:" << std::endl;
  std::cout << " num of steps: " <<  num_steps << std::endl;
  std::cout << " step size:" << epsilon << std::endl;

  
  //// read files
  std::cout << std::endl << "////////////////////////////////////////////////////////////" << std::endl;  
  std::cout << "Reading data and make Neural Netowrk objects for training and validation" << std::endl;
  std::ifstream in_file;  
  int train_batch_size = 10000;
  NN train_nn(num_nodes.size(), num_nodes, train_batch_size);

  // read train data
  int train_data_size = 50000;
  in_file.open("./data/train_image.txt", std::ifstream::in);
  float* train_image = new float[train_data_size*dim];
  for(int i = 0; i < train_data_size; i++)
  {
    for(int j = 0; j < dim; j++)
    {
      in_file >> train_image[IDX(i,j,train_data_size)];
    }
  }
  in_file.close();
  
  // train label
  int *train_label;
  train_label = new int[train_data_size];
  in_file.open("./data/train_label.txt", std::ifstream::in);
  for(int i = 0; i < train_data_size; i++)
  {
    in_file >> train_label[i];
  }
  in_file.close();

  // validation data set
  int validation_data_size = 10000;
  NN validation_nn(num_nodes.size(), num_nodes, validation_data_size);

  // read validation data
  in_file.open("./data/validation_image.txt", std::ifstream::in);
  float* validation_image = new float[validation_data_size*dim];
  for(int i = 0; i < validation_data_size; i++)
  {
    for(int j = 0; j < dim; j++)
    {
      in_file >> validation_image[IDX(i,j,validation_data_size)];
    }
  }
  in_file.close();
  
  // validation label
  int *validation_label;
  validation_label = new int[validation_data_size];
  in_file.open("./data/validation_label.txt", std::ifstream::in);
  for(int i = 0; i < validation_data_size; i++)
  {
    in_file >> validation_label[i];
  }
  in_file.close();
  validation_nn.set_batch(validation_image, validation_label, validation_data_size, dim);

  // random select a batch of data and label from training data
  float* train_batch_image;
  int* train_batch_label;
  train_batch_image = new float [train_batch_size*dim];
  train_batch_label = new int [train_batch_size];
  std::vector<int> idxs;
  for(int i = 0; i < train_data_size; i++)
  {
    idxs.push_back(i);
  }

  //// Start the training process
  std::cout << std::endl << "////////////////////////////////////////////////////////////" << std::endl;  
  std::cout << "Starint training ......" << std::endl;
  for(int n = 0; n < num_steps; n++)
  {
    std::random_shuffle(idxs.begin(), idxs.end());  
    for(int i = 0; i < train_batch_size; i++)
    {
      for(int j = 0; j < dim; j++)
      {
	train_batch_image[IDX(i,j,train_batch_size)] = train_image[IDX(idxs[i],j,train_data_size)];
      }
      train_batch_label[i] = train_label[idxs[i]];
    }
    train_nn.set_batch(train_batch_image, train_batch_label, train_batch_size, dim);
    float cost = train_nn.train(1, epsilon);  
    validation_nn.set_weights(train_nn.get_weights());
    validation_nn.set_bias(train_nn.get_bias());
    validation_nn.feed_forward();
    std::cout << "Step: " << std::setw(4) << std::right << n << ", ";
    std::cout << "Cost: " << std::fixed << std::setprecision(3) << cost << ", ";
    std::cout << "training accuracy: " <<  train_nn.calc_accuracy() << ", ";
    std::cout << "validation accuracy: " <<  validation_nn.calc_accuracy() << std::endl;
  }

  std::cout << std::endl << "////////////////////////////////////////////////////////////" << std::endl;
  std::cout << "Evalution the model on testing data:" << std::endl;
  std::cout << "Reading data and make Neural Netowrk objects for testing" << std::endl;  
  // read test image
  int test_data_size = 10000;
  in_file.open("./data/test_image.txt", std::ifstream::in);
  float* test_image = new float[test_data_size*dim];
  for(int i = 0; i < test_data_size; i++)
  {
    for(int j = 0; j < dim; j++)
    {
      in_file >> test_image[IDX(i,j,test_data_size)];
    }
  }
  in_file.close();
  
  // test label
  int *test_label;
  test_label = new int[test_data_size];
  in_file.open("./data/test_label.txt", std::ifstream::in);
  for(int i = 0; i < test_data_size; i++)
  {
    in_file >> test_label[i];
  }
  in_file.close();
  
  NN test_nn(num_nodes.size(), num_nodes, test_data_size);
  test_nn.set_batch(test_image, test_label, test_data_size, dim);  
  test_nn.set_weights(train_nn.get_weights());
  test_nn.set_bias(train_nn.get_bias());
  test_nn.feed_forward();
  std::cout << "Cost: " << std::fixed << std::setprecision(3) << test_nn.calc_cost() << ", ";
  std::cout << "test accuracy: " <<  test_nn.calc_accuracy() << std::endl;
  
  return 0;
}
