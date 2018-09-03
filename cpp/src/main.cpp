#include "DataLoader.hpp"
#include "WordBeamSearch.hpp"
#include "Metrics.hpp"
#include "test.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <algorithm>


// run unit tests: uncomment next line and run in debug mode
//#define UNITTESTS 


int main()
{
	std::ofstream f("dump.txt");

#ifdef UNITTESTS
	test();
#else
	const std::string baseDir = "/home/mmedina/WindowsDesktop/CSV_samples/"; // dir containing corpus.txt, chars.txt, wordChars.txt, mat_x.csv, gt_x.txt with x=0, 1, ...
	//const std::string baseDir = "../data/utf8test/"; // dir containing corpus.txt, chars.txt, wordChars.txt, mat_x.csv, gt_x.txt with x=0, 1, ...
	const size_t sampleEach = 1; // only take each k*sampleEach sample from dataset, with k=0, 1, ...
	const double addK = 1.0; // add-k smoothing of bigram distribution
	const LanguageModelType lmType = LanguageModelType::Words; // scoring mode
	DataLoader loader{ baseDir, sampleEach, lmType, addK }; // load data
	const auto& lm = loader.getLanguageModel(); // get LM
	Metrics metrics{ lm->getWordChars() }; // CER and WER

	const std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
	size_t ctr = 0;
	while (loader.hasNext())
	{
		// get data
		const auto&  data=loader.getNext();

		// decode it
		const auto res = wordBeamSearch(data.mat, 1, lm, lmType);

		// show results
		std::cout << "Sample: " << ctr + 1 << "\n";


		std::string firstDecoded;
		for (size_t i=0; i<res.size(); ++i)
		{
		  std::string decoded = lm->labelToUtf8(res[i]);
		  decoded.erase(remove_if(decoded.begin(), decoded.end(), isspace), decoded.end());

		  if (i == 0)
		    firstDecoded = decoded;

		  std::cout << "Beam #"<<i<<":       \"" << decoded << "\"\n";
			f << decoded << "\n";
		}
		

		std::string gt = lm->labelToUtf8(data.gt);
		gt.erase(remove_if(gt.begin(), gt.end(), isspace), gt.end());
		std::cout << "Ground Truth: \"" << gt << "\"\n";
		metrics.addResult(gt, firstDecoded);
		std::cout << "Accumulated CER and Accuracy so far: CER: " << metrics.getCER() << " Accuracy " << metrics.getAccuracy() << "\n";
		const std::chrono::system_clock::time_point currTime = std::chrono::system_clock::now();
		std::cout << "Average Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(currTime-startTime).count()/(ctr+1) << "ms\n\n";
		++ctr;
	}

	std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - startTime).count() << "ms\n";
#endif

	std::cout<<"Press any key to continue\n";
	getchar();
	return 0;
}
