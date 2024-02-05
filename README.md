main.cpp contains whole code of layer wise operations of cnn_cifar_model applied on image and filters.
input image, weights(layer-wise),outputs(layer-wise) are downloaded from py files.
cpp code output and py code output are checked and exceution time is also tracked.
config file contains information about layers and  related shapes, this is read in main code, to perform sequential operations like conv followed by relu,pooling,flatten and dense
