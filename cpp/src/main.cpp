#include "DataLoader.hpp"
#include "WordBeamSearch.hpp"
#include "Metrics.hpp"
#include "test.hpp"
#include <iostream>
#include <chrono>
#include <fstream>



// run unit tests: uncomment next line and run in debug mode
//#define UNITTESTS 


int main()
{

#ifdef UNITTESTS
	test();
#else
	const std::string baseDir = "/home/mmedina/WindowsDesktop/CSV_samples/";
	//const std::string baseDir = "../../data/bentham/"; // dir containing corpus.txt, chars.txt, wordChars.txt, mat_x.csv, gt_x.txt with x=0, 1, ...
	const size_t sampleEach = 1; // only take each k*sampleEach sample from dataset, with k=0, 1, ...
	const double addK = 1.0; // add-k smoothing of bigram distribution
	const LanguageModelType lmType = LanguageModelType::Words; // scoring mode
	DataLoader loader{ baseDir, sampleEach, lmType, addK }; // load data
	const auto& lm = loader.getLanguageModel(); // get LM
	Metrics metrics{ lm->getWordChars() }; // CER and WER

	const std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
	size_t ctr = 0;
	size_t nBestBeams = 10;
	size_t n_sample = 0;

	
	while (loader.hasNext())
	{
	  std::cout << "Sample: " << ctr + 1 << "\n";
	  
	  // get data
	  const auto&  data=loader.getNext();
	  
	  std::cout << "Got data...." << std::endl;
	  
	  // decode it
	  const auto res = wordBeamSearch(data.mat, nBestBeams, 10, lm, lmType);
	  
	  std::cout << "Decoded it... Size: " << res.size() << std::endl;
	  
	  std::string fname = "/home/mmedina/WindowsDesktop/WBS_Decoded/" + std::to_string(n_sample++) + ".csv";
	  
	  std::ofstream outFile(fname);
	  
	  // show results
	  for (size_t i = 0; i < nBestBeams; i++)
	    {
	      
	      for (size_t j = 0; j < res[i].size(); j++)
		outFile << res[i][j] << ",";
	      
	      outFile << "\n";
	      //std::cout << j + 1 << ": " << res[i][j] << std::endl;
	      
	      /*std::cout << "Result " << i + 1 << std::endl;
		std::cout << "Decoded:       \"" << lm->labelToUtf8(res[i]) << "\"\n";
		std::cout << "Size: " << res[i].size() << std::endl;*/
	    }
	  
	  
	  
	  std::cout << "Saved!" << std::endl;
	  
	  //std::cout << "Ground Truth: \"" << lm->labelToUtf8(data.gt) << "\"\n";
	  /*metrics.addResult(data.gt, res[i]);
	    std::cout << "Accumulated CER and WER so far: CER: " << metrics.getCER() << " WER: " << metrics.getWER() << "\n";*/
	  const std::chrono::system_clock::time_point currTime = std::chrono::system_clock::now();
	  std::cout << "Average Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(currTime-startTime).count()/(ctr+1) << "ms\n\n";

	  
	  
	  ++ctr;
	  
	}
#endif

	std::cout<<"Press any key to continue\n";
	getchar();
	return 0;
}
