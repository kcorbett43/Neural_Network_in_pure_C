CC = nvcc

CFLAGS = -O3 

LDFLAGS = -lm -lgsl -lgslcblas -lcublas -DHAVE_CUBLAS

SRC = gpuBlas.cu
OUT = gpuBlas

all: $(OUT)

$(OUT): $(SRC)
	$(CC) $(SRC) -o $(OUT) $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(OUT)
