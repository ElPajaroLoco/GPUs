CXX=icpx
CXXFLAGS=-fsycl -I.
LIBS=-L/usr/lib/gcc/x86_64-linux-gnu/11

%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

exec: mult_matrix.o
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LIBS) 

clean:
	rm -f exec *.o
