
#include <iostream>
#include <armadillo>
class MaxPoolingLayer
{
 public:
  MaxPoolingLayer(size_t inputHeight,
                  size_t inputWidth,
                  size_t inputDepth,
                  size_t poolingWindowHeight,
                  size_t poolingWindowWidth,
                  size_t verticalStride,
                  size_t horizontalStride) :
      inputHeight(inputHeight),
      inputWidth(inputWidth),
      inputDepth(inputDepth),
      poolingWindowHeight(poolingWindowHeight),
      poolingWindowWidth(poolingWindowWidth),
      verticalStride(verticalStride),
      horizontalStride(horizontalStride)
  {
    // Nothing to do here.
  }

  void Forward(arma::cube& input, arma::cube& output)
  {
    assert((inputHeight - poolingWindowHeight)%verticalStride == 0);
    assert((inputWidth - poolingWindowWidth)%horizontalStride == 0);
    output = arma::zeros(
        (inputHeight - poolingWindowHeight)/verticalStride + 1,
        (inputWidth - poolingWindowWidth)/horizontalStride + 1,
        inputDepth
        );
    for (size_t sidx = 0; sidx < inputDepth; sidx ++)
    {
      for (size_t ridx = 0;
           ridx <= inputHeight - poolingWindowHeight;
           ridx += verticalStride)
      {
        for (size_t cidx = 0;
             cidx <= inputWidth - poolingWindowWidth;
             cidx += horizontalStride)
        {
          output.slice(sidx)(ridx/verticalStride, cidx/horizontalStride) =
            input.slice(sidx).submat(ridx,
                          cidx,
                          ridx+poolingWindowHeight-1,
                          cidx+poolingWindowWidth-1)
            .max();
        }
      }
    }

    this->input = input;
    this->output = output;

  }

  

 private:
  size_t inputHeight;
  size_t inputWidth;
  size_t inputDepth;
  size_t poolingWindowHeight;
  size_t poolingWindowWidth;
  size_t verticalStride;
  size_t horizontalStride;

  arma::cube input;
  arma::cube output;

};

