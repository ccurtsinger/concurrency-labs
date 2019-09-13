CC := nvcc
CFLAGS := -g

all: sudoku inputs
	
clean:
	rm -f sudoku

sudoku: sudoku.cu util.h Makefile
	$(CC) $(CFLAGS) -o sudoku sudoku.cu

inputs: inputs.tar.gz
	tar xvzf inputs.tar.gz
