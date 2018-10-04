OPTIONS = -Wall -O2 --std=c++11
COMPILER = g++
FILES =  ../src/main.cpp ../src/WordBeamSearch.cpp ../src/PrefixTree.cpp ../src/Metrics.cpp ../src/MatrixCSV.cpp ../src/LanguageModel.cpp ../src/DataLoader.cpp ../src/Beam.cpp
TARGET = WordBeamSearch.out
TENSORFLOW_TARGET = TFWordBeamSearch.so
TENSORFLOW_OPTIONS = -shared
TF_CFLAGS = ( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=  ( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
TENSORFLOW_EXTRA_OPTIONS = -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -D_GLIBCXX_USE_CXX11_ABI=0

wbsd : clean
	$(COMPILER) $(OPTIONS) -o $(TARGET) $(FILES)

shared : clean_tf
	$(COMPILER) $(OPTIONS) $(TENSORFLOW_OPTIONS) -o $(TENSORFLOW_TARGET) $(FILES) $(TENSORFLOW_EXTRA_OPTIONS)


.PHONY : clean_all

clean_all : clean clean_tf

.PHONY : clean

clean :
	-rm $(TARGET)

.PHONY : clean_tf

clean_tf :
	-rm $(TENSORFLOW_TARGET)

