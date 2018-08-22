#include "MatrixNP.hpp"
#include <boost/python.hpp>

MatrixNP::MatrixNP(pyublas::numpy_vector<double> npArray, size_t bNumber);
{
  npy_intp tSteps = npArray.dims()[1];
  npy_intp nClasses = npArray.dims()[2];

  
  for (npy_intp t = 0; t < tSteps; t++)      
    for (npy_intp c = 0; c < nClasses; c++)	    
      m_data[bNumber][t][c] = npArray.sub(b,t,c);


  m_rows = tSteps;
  m_cols = nClasses;
}

double MatrixNP::getAt(size_t row, size_t col)
{
  return m_data[row][col];
}

double MatrixNP::setAt(size_t row, size_t col, double val)
{
  m_data[row][col] = val;
}
