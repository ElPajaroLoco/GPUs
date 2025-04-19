CXX=icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda 
CXXFLAGS= -I.
LIBS=

%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

exec: main.o
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LIBS) 

clean:
	rm -f exec *.o
