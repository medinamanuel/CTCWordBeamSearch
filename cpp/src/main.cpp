#include "DataLoader.hpp"
#include "WordBeamSearch.hpp"
#include "Metrics.hpp"
#include "test.hpp"
#include <iostream>
#include <chrono>
#include <string>
#include <algorithm>


// run unit tests: uncomment next line and run in debug mode
//#define UNITTESTS 


int main()
{

#ifdef UNITTESTS
	test();
#else
	//const std::string baseDir = "../../data/bentham/"; // dir containing corpus.txt, chars.txt, wordChars.txt, mat_x.csv, gt_x.txt with x=0, 1, ...
	const std::string baseDir = "/home/mmedina/WindowsDesktop/CSV_samples/"; // dir containing corpus.txt, chars.txt, wordChars.txt, mat_x.csv, gt_x.txt with x=0, 1, ...
	const size_t sampleEach = 1; // only take each k*sampleEach sample from dataset, with k=0, 1, ...
	const double addK = 1.0; // add-k smoothing of bigram distribution
	const LanguageModelType lmType = LanguageModelType::Words; // scoring mode
	std::cout << "Loading data from " << baseDir << std::endl;
	DataLoader loader{ baseDir, sampleEach, lmType, addK }; // load data
	const auto& lm = loader.getLanguageModel(); // get LM
	Metrics metrics{ lm->getWordChars() }; // CER and WER

	const std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
	size_t ctr = 0;
	size_t nBestBeams = 1;
	while (loader.hasNext())
	{
		// get data
		const auto&  data=loader.getNext();

		// decode it
		const auto res = wordBeamSearch(data.mat, nBestBeams, 10, lm, lmType);

		std::cout << "Sample: " << ctr + 1 << "\n";

		// show results

		std::string decoded = lm->labelToUtf8(res[0]);
		std::string gt = lm->labelToUtf8(data.gt);
		
		decoded.erase(remove_if(decoded.begin(), decoded.end(), isspace), decoded.end());
		gt.erase(remove_if(gt.begin(), gt.end(), isspace), gt.end());

		std::cout << "Result:       \"" << decoded << "\"\n";
		std::cout << "Ground Truth: \"" << gt << "\"\n";		
		metrics.addResult(gt, decoded);
		std::cout << "Accumulated CER and Accuracy so far: CER: " << metrics.getCER() << " Accuracy: " << metrics.getAccuracy() << "\n";
		const std::chrono::system_clock::time_point currTime = std::chrono::system_clock::now();
		std::cout << "Average Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(currTime-startTime).count()/(ctr+1) << "ms\n\n";
		
		++ctr;
		
	}
#endif

	std::cout<<"Press any key to continue\n";
	getchar();
	return 0;
}
