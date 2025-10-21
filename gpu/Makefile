CC = nvcc

CFLAGS = -O3 

LDFLAGS = -lm -lgsl -lgslcblas 

SRC = gpu.cu
OUT = gpu

all: $(OUT)

$(OUT): $(SRC)
	$(CC) $(SRC) -o $(OUT) $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(OUT)
