# A CUDA Parallel Implementation of Feed Forward Neural Networks for MNIST Recognition on GPU

## Description
This is an implementation of feed forward neural networks for recogniting the MNIST. It is written in C++ with CUDA. 
The feed forward neural network comprises one input layer with 28X28 neurons, several hidden layers and a softmax layer.
In the main.cpp file, you can specify the number of hidden layers and the number of neurons in each hidden layer.

## How to Install
### Required Library
* CUDA(7.0)
* GCC(4.8.4 or newer)

### Compile
After cloning the repository, change the directory to **NeuralNetwork\_GPU\_MNIST**, in which there is a **Makefile**. Type ```make```. The ```make``` command will compile all the object files and the main exectable file inside directory **build**. At the same time, it also copies the main exectable file into directory **test**. Therefore, after the compiling, you should have an exectable file called **main** inside the directory **test**.






<!-- 1. load the following module before make -->
<!-- cuda/7.0 -->
<!-- gcc/4.8.4 -->
