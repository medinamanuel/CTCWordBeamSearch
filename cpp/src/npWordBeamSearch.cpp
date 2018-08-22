#include <boost/python.hpp>
#include "npWordBeamSearch.hpp"
#include "LanguageModel.hpp"
#include "WordBeamSearch.hpp"

using namespace std

typedef BeamSearchResults vector<vector<uint32_t>>;

LanguageModelType strToLMType(string strLmType)
{
  transform(strLmType.begin(), strLmType.end(), strLmType.begin(), ::tolower);
  LanguageModelType lmType;

  if (strLmType == "words")
    lmType = LanguageModelType::Words;
  else if (strLmType == "ngrams")
    lmType = LanguageModelType::NGrams;
  else if (strLmType == "ngramsforecast")
    lmType = LanguageModelType::NGramsForecast;
  else if (strLmType == "ngramsforecastandsample")
    lmType = LanguageModelType::NGramsForecastAndSample;
  else
    throw invalid_argument("Unknown Language Model Type");

  return lmType;
}


pyublas::numpy_vector<double> npWordBeamSearch(pyublas::numpy_vector<double> npArray, size_t nBestBeams, size_t beamWidth, string lm_type, float lmSmoothing, string corpus, string chars, string wordChars)
{
  std::vector<BeamSearchResults> beamsByBatches;
  LanguageModelType lmType = strtoLMType(lm_type);

  npy_intp nBatches = npArray.dims()[0];
  npy_intp tSteps = npArray.dims()[1];

  // Create Language Model
  shared_ptr<LanguageModel> lm = make_shared<LanguageModel>(corpus, chars, wordChars, lmType, lmSmoothing);

  // Return value
  int nDims = 3;
  npy_intp dims[nDims];
  dims[0] = nBatches;
  dims[1] = nBestBeams;
  dims[2] = tSteps;
  
  pyublass::numpy_vector<npy_intp> results = pyublas::numpy_vector<npy_intp>(nDims, dims);

  // Word Beam Search to each batch
  for (npy_intp b = 0; b < nBatches; b++)
    {
      MatrixNP nextBatch = MatrixNP(npArray, b);

      // r is (nBestBeams, tSteps)
      BeamSearchResults r = wordBeamSearch(nextBatch, nBestBeams, beamWidth, lm, lmType);

      for (size_t nbb = 0; nbb < nBestBeams; n++)
	for(size_t ts = 0; ts < tSteps; t++)
	  {
	    results[b + (nbb * tSteps) + ts] = r[nbb][ts];
	  }
    }

  // Build the NP Array to return

}
