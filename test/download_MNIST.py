__author__ = "Xinqiang Ding <xqding@umich.edu>"

import numpy as np
import urllib3
import gzip
import subprocess

subprocess.run(["mkdir", '-p', './data/'])

## download train labels
print("Downloading train-labels-idx1-ubyte ......")
http = urllib3.PoolManager()
r = http.request('GET', 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
data = gzip.decompress(r.data)
num = int.from_bytes(data[4:8], 'big')
offset = 8
label = np.array([data[offset+i] for i in range(num)])

## download train image
print("Downloading train-image-idx3-ubyte ......")
http.clear()
r = http.request('GET', 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
data = gzip.decompress(r.data)
num = int.from_bytes(data[4:8], 'big')
nrows = int.from_bytes(data[8:12], 'big')
ncols = int.from_bytes(data[12:16], 'big')
image = np.zeros((num, nrows * ncols))
offset = 16
for k in range(num):
    for i in range(nrows):
        for j in range(ncols):
            image[k, i*ncols+j] = data[16 + k*nrows*ncols + i*ncols+j]
image = image / 255.0
            
np.random.seed(0)
num_train = 50000
num_validation = num - num_train
train_idx = np.random.choice(range(num), size = num_train, replace = False)
validation_idx = list(set(range(num)) - set(train_idx))

train_image = image[train_idx, :]
validation_image = image[validation_idx, :]
train_label = label[train_idx]
validation_label = label[validation_idx]

## download test labels
print("Downloading t10k-labels-idx1-ubyte ......")
http.clear()
r = http.request('GET', 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')
data = gzip.decompress(r.data)
num = int.from_bytes(data[4:8], 'big')
offset = 8
test_label = np.array([data[offset+i] for i in range(num)])

## download test image
print("Downloading t10k-image-idx3-ubyte ......")
http.clear()
r = http.request('GET', 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
data = gzip.decompress(r.data)
num = int.from_bytes(data[4:8], 'big')
nrows = int.from_bytes(data[8:12], 'big')
ncols = int.from_bytes(data[12:16], 'big')
test_image = np.zeros((num, nrows * ncols))
offset = 16
for k in range(num):
    for i in range(nrows):
        for j in range(ncols):
            test_image[k, i*ncols+j] = data[16 + k*nrows*ncols + i*ncols+j]
            
test_image = test_image / 255.0

print("Saving data into txt files ...")            
np.savetxt("./data/train_image.txt", train_image, fmt = "%f")
np.savetxt("./data/validation_image.txt", validation_image, fmt = "%f")
np.savetxt("./data/test_image.txt", test_image, fmt = "%f")
np.savetxt("./data/train_label.txt", train_label, fmt = "%d")
np.savetxt("./data/validation_label.txt", validation_label, fmt = "%d")
np.savetxt("./data/test_label.txt", test_label, fmt = "%d")
