#pragma once
#include "MatrixNP.hpp"
#include <pyublas/numpy.hpp>
#include "LanguageModel.hpp"


LanguageModelType strToLMType(std::string strLmType);

pyublas::numpy_vector<double> npWordBeamSearch(pyublas::numpy_vector<double> npArray, size_t nBestBeams, size_t beamWidth, string lmType, float lmSmoothing, string corpus, string chars, string wordChars);
