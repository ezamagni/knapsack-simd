CC = nvcc
BIN = gpuknapsack
CCFLAGS = -arch=sm_11

all:$(BIN)

gpuknapsack:
	$(CC) -o $@ $(CCFLAGS) src/gpuknapsack.cu

clean:
	rm -f *.o *.*~ $(BIN)
