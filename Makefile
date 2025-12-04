CC = nvcc
CFLAG = -std=c++17 -O3 -Xcompiler -fopenmp
SRC = hw3-2.cu
OBJ = hw3-2

$(OBJ):
	$(CC) $(CFLAG) $(SRC) -o $(OBJ)

clean: $(OBJ)
	rm -rf $(OBJ)
