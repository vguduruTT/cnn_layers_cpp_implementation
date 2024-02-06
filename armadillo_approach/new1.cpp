#include <iostream>
#include "npy.hpp"
#include <bits/stdc++.h>
#include <armadillo>
using namespace std;
using namespace arma;
float fun(arma::mat x1,arma::mat x2)
{
    float sum=0;
    for(int i=0;i<x1.n_cols;i++)
    {
        for(int j=0;j<x1.n_rows;j++)
        {
            sum = sum  + (x1(i,j)*x2(i,j));
        }
    }
    return sum;
    // return arma::accu(x1 % x2);
}



std::vector<arma::mat> convolution(const arma::cube& input, const std::vector<arma::cube>& filters, size_t verticalStride, size_t horizontalStride,std::vector<float>biases) {
    // Get dimensions of input image and filters
    size_t inputHeight = input.n_rows;
    size_t inputWidth = input.n_cols;
    size_t inputChannels = input.n_slices;
    size_t numFilters = filters.size();
    size_t filterHeight = filters[0].n_rows; // Assuming all filters have the same dimensions
    size_t filterWidth = filters[0].n_cols;

    // Compute output dimensions
    size_t outputHeight = (inputHeight - filterHeight) / verticalStride + 1;
    size_t outputWidth = (inputWidth - filterWidth) / horizontalStride + 1;

    // Define vector of output matrices to store the result for each filter
    std::vector<arma::mat> output(numFilters, arma::mat(outputHeight, outputWidth, arma::fill::zeros));

    // Convolution operation
    for (size_t f = 0; f < numFilters; ++f) { // Iterate over filters
        for (size_t i = 0; i <= inputHeight - filterHeight; i += verticalStride) { // Iterate vertically
            for (size_t j = 0; j <= inputWidth - filterWidth; j += horizontalStride) { // Iterate horizontally
                // Reset the dot product sum for this position
                float dotProductSum = 0.0;

                for (size_t ch = 0; ch < inputChannels; ++ch) { // Iterate over channels
                    // Extract the current slice from the filter
                    arma::mat currentFilterSlice = filters[f].slice(ch);
                    //  cout << currentFilterSlice << endl;
                    // Extract the corresponding slice from the image
                    arma::mat currentImageSlice = input.slice(ch).submat(i, j, i + filterHeight - 1, j + filterWidth - 1);
                    // cout << currentImageSlice << endl;
                    // Compute the dot product of the current slices and add it to the dot product sum
                    // cout << fun(currentFilterSlice,currentImageSlice) <<" " << endl;
                          dotProductSum= dotProductSum+ fun(currentFilterSlice,currentImageSlice);
                    // dotProductSum += arma::accu(currentImageSlice % currentFilterSlice);
                }

                // Assign the dot product sum to the output matrix
                output[f](i / verticalStride, j / horizontalStride) += dotProductSum + biases[f];
            }

        }
    }

    return output;
}
int main() {
    // // Load the .npy file
    const std::string path{"in_torch.npy"};
    npy::npy_data<float> d = npy::read_npy<float>(path);
    std::vector<float> dat = d.data;
    cube c(32, 32, 3);
    int kk=0;
    for(int q=0;q<3;q++)
    {
    arma :: mat x(32,32);
    
    for (arma::uword i = 0; i < x.n_rows; ++i) {
    for (arma::uword j = 0; j < x.n_cols; ++j) {
        x(i,j)=dat[kk++];
        
    }
    } 
    c.slice(q)=x;
    }
    // std::cout << c.slice(2);
    // std::cout << c(0,0,0) << std::endl;
    // std:: cout << c.size();
    //weights
        size_t filterHeight = 3;
    size_t filterWidth = 3;
    size_t filterChannels = 3;
    size_t numFilters = 3;
    size_t verticalStride = 1;
    size_t horizontalStride = 1;
            const std::string path2{"1_w_1.npy"};
    npy::npy_data<float> df = npy::read_npy<float>(path2);
    std::vector<float> dat2 = df.data;
    // std:: cout << dat2[0]  << "\n";
        size_t tfs = filterChannels*filterHeight*filterWidth;
       std::vector<arma::cube> filters;
        filters.resize(numFilters);
    for (size_t i = 0; i < numFilters; i++) {
        filters[i] = arma::zeros(filterHeight, filterWidth,filterChannels);
    }
    kk=0;
    for (size_t i = 0; i < numFilters; i++) {
       arma::cube tmp(3,3,3);
          for (size_t k = 0; k < tmp.n_slices; ++k) {
        for (size_t i = 0; i < tmp.n_rows; ++i) {
            for (size_t j = 0; j < tmp.n_cols; ++j) {
                tmp(i, j, k) = dat2[kk++];
            }
        }
    }
       filters[i]=tmp;
    }
     const std::string path3{"1_b_1.npy"};
    npy::npy_data<float> dfg = npy::read_npy<float>(path3);
    std::vector<float> dat3 = dfg.data;
    vector<arma::mat> output=convolution(c,filters,1,1,dat3);
    
    return 0;
  
}
