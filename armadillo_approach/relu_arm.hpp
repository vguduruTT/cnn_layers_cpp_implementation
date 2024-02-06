
#include <iostream>
#include <armadillo>

class ReLULayer
{
 public:
  ReLULayer(size_t inputHeight,
       size_t inputWidth,
       size_t inputDepth) :
      inputHeight(inputHeight),
      inputWidth(inputWidth),
      inputDepth(inputDepth)
  {
    // Nothing to do here.
  }

  void Forward(arma::cube& input, arma::cube& output)
  {
    output = arma::zeros(arma::size(input));
    output = arma::max(input, output);
    this->input = input;
    this->output = output;
  }

  

 private:
  size_t inputHeight;
  size_t inputWidth;
  size_t inputDepth;

  arma::cube input;
  arma::cube output;


};


