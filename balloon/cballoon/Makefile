CC = clang-omp
DEPS = balloon.h mergesort.h
OBJ = balloon.o test_correctness.o test_time.o mergesort.o
CCFLAGS = -Wall -fopenmp -fpic

%.o: %.c $(DEPS)
	$(CC) $(CCFLAGS) -c -o $@ $<

test_correctness: test_correctness.o balloon.o mergesort.o
	$(CC) $(CCFLAGS) -o $@ $^ 

test_time: test_time.o balloon.o mergesort.o
	$(CC) $(CCFLAGS) -o $@ $^ 

shared: balloon.o mergesort.o
	$(CC) $(CCFLAGS) -shared -o balloon.so $^

clean:
	rm -f *.o *~

