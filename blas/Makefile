CC = gcc

CFLAGS = -O3 -ffast-math

LDFLAGS = -lm -lgsl -lgslcblas -lopenblas -DHAVE_OPENBLAS

SRC = cpuBlas.c
OUT = cpuBlas

all: $(OUT)

$(OUT): $(SRC)
	$(CC) $(SRC) -o $(OUT) $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(OUT)
