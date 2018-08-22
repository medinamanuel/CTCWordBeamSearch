#pragma once
#include "IMatrix.hpp"
#include <pyublas/numpy.hpp>


class MatrixNP : public IMatrix
{
  typedef md_vector std::vector<std::vector<double>>;
  
public:
  explicit MatrixNP(pyublas::numpy_vector<double> npArray, size_t bNumber);

  virtual double getAt(size_t row, size_t col) const;
  virtual void setAt(size_t row, size_t col, double val);
  size_t rows() const { return m_rows; }
  size_t cols() const { return m_cols; }


private:
  md_vector m_data;
}


