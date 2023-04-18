INC_FLAGS := $(addprefix -I,inc/)

all: fast-fir

filter_parse.o: filter_parse.cpp filter_parse.h
	g++ -Wall -c filter_parse.cpp

wav_parse.o: wav_parse.cpp wav_parse.h
	g++ -Wall -c wav_parse.cpp

fast-fir: fast-fir.cu filter_parse.o wav_parse.o
	nvcc $^ -o $@ -lcufft $(INC_FLAGS)

clean:
	rm -f fast-fir wav_parse.o filter_parse.o
