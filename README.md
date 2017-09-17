# A CUDA Parallel Implementation of Feed Forward Neural Networks for MNIST Recognition on GPU

## Description
This is an implementation of feed forward neural networks for recogniting the MNIST. It is written in C++ with CUDA. 
The feed forward neural network comprises one input layer with 28X28 neurons, several hidden layers and a softmax layer.
In the main.cpp file, you can specify the number of hidden layers and the number of neurons in each hidden layer.

## How to Install
### Required library
* CUDA(7.0)
* GCC(4.8.4 or newer)

### Compile
After cloning the repository, change the directory to **NeuralNetwork\_GPU\_MNIST**, in which there is a **Makefile**. Type ```make```. The ```make``` command will compile all the object files and the main exectable file inside directory **build**. At the same time, it also copies the main exectable file into directory **test**. Therefore, after the compiling, you should have an exectable file called **main** inside the directory **test**.

## Train and Test the Neural Network Model
1. **Download the MNIST dataset**

   The MNIST dataset is a collection of 28 X 28 images of handwritten digits. The data set is hosted [here](http://yann.lecun.com/exdb/mnist/). The Python3 script **download_MNIST.py** inside the directory **test** can be used to download and format the MNIST dataset using the following command:  
   ```
   cd ./test
   python download_MNIST.py
   ```  
   The command generates the following data file inside **test/data** directory:  
   ```
   train_image.txt train_label.txt
   validation_image.txt validation_label.txt
   test_image.txt test_label.txt.
   ```  
   The Python3 script **download_MNIST.py** needs packages: **numpy, urllib3, gzip**, and **subprocess**.

2. **Train and test the model**

   Run the following command:  
   ```
   cd ./test/
   ./main
   ```
