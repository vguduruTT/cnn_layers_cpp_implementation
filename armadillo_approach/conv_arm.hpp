
#include <armadillo>
#include <vector>
#include <iostream>
#include <cmath>
class ConvolutionLayer
{
 public:
  ConvolutionLayer(
      size_t inputHeight,
      size_t inputWidth,
      size_t inputDepth,
      size_t filterHeight,
      size_t filterWidth,
      size_t horizontalStride,
      size_t verticalStride,
      size_t numFilters) :
    inputHeight(inputHeight),
    inputWidth(inputWidth),
    inputDepth(inputDepth),
    filterHeight(filterHeight),
    filterWidth(filterWidth),
    horizontalStride(horizontalStride),
    verticalStride(verticalStride),
    numFilters(numFilters)
  {
    // Initialize the filters.
    filters.resize(numFilters);
    for (size_t i=0; i<numFilters; i++)
    {
      filters[i] = arma::zeros(filterHeight, filterWidth, inputDepth);
      filters[i].imbue( [&]() { return _getTruncNormalVal(0.0, 1.0); } );
    }
    //reseting gradients for next round of back propag

  }

  void Forward(arma::cube& input, arma::cube& output)
  {
    // The filter dimensions and strides must satisfy some contraints for
    // the convolution operation to be well defined.
    assert((inputHeight - filterHeight)%verticalStride == 0);
    assert((inputWidth - filterWidth)%horizontalStride == 0);

    // Output initialization.
    output = arma::zeros((inputHeight - filterHeight)/verticalStride + 1,
                         (inputWidth - filterWidth)/horizontalStride + 1,
                         numFilters);

    // Perform convolution for each filter.
    for (size_t fidx = 0; fidx < numFilters; fidx++)
    {
      for (size_t i=0; i <= inputHeight - filterHeight; i += verticalStride)
        for (size_t j=0; j <= inputWidth - filterWidth; j += horizontalStride)
          output((i/verticalStride), (j/horizontalStride), fidx) = arma::dot(
              arma::vectorise(
                  input.subcube(i, j, 0,
                                i+filterHeight-1, j+filterWidth-1, inputDepth-1)
                ),
              arma::vectorise(filters[fidx]));
    }

    // Store the input and output. This will be needed by the backward pass.
    this->input = input;
    this->output = output;


  }
 private:
  size_t inputHeight;
  size_t inputWidth;
  size_t inputDepth;
  size_t filterHeight;
  size_t filterWidth;
  size_t horizontalStride;
  size_t verticalStride;
  size_t numFilters;
  std::vector<arma::cube> filters;
  arma::cube input;
  arma::cube output;

};
