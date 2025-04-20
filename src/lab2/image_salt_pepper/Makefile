
CXX = icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda

CXXFLAGS= -O3

LIBS=


main: main.cpp kernels.cpp 
	$(CXX) $^ $(CXXFLAGS) $(LIBS) -o $@


clean:
	rm -f main
