
#include <armadillo>
#include <vector>
#include <cmath>
class DenseLayer
{
 public:
  DenseLayer(size_t inputHeight,
             size_t inputWidth,
             size_t inputDepth,
             size_t numOutputs) :
      inputHeight(inputHeight),
      inputWidth(inputWidth),
      inputDepth(inputDepth),
      numOutputs(numOutputs)
  {
    // Initialize the weights.
    weights = arma::zeros(numOutputs, inputHeight*inputWidth*inputDepth);
    weights.imbue( [&]() { return _getTruncNormalVal(0.0, 1.0); } );

    // Initialize the biases
    biases = arma::zeros(numOutputs);

    // Reset accumulated gradients.
    _resetAccumulatedGradients();
  }

  void Forward(arma::cube& input, arma::vec& output)
  {
    arma::vec flatInput = arma::vectorise(input);
    output = (weights * flatInput) + biases;

    this->input = input;
    this->output = output;
  }

  
 private:
  size_t inputHeight;
  size_t inputWidth;
  size_t inputDepth;
  arma::cube input;

  size_t numOutputs;
  arma::vec output;

  arma::mat weights;
  arma::vec biases;



  double _getTruncNormalVal(double mean, double variance)
  {
    double stddev = sqrt(variance);
    arma::mat candidate = {3.0 * stddev};
    while (std::abs(candidate[0] - mean) > 2.0 * stddev)
      candidate.randn(1, 1);
    return candidate[0];
  }

 
};


