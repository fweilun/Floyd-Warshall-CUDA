CC = nvcc
CFLAG = -std=c++17 -O3
SRC = hw3-2.cu
OBJ = hw3-2

$(OBJ):
	$(CC) $(CFLAG) $(SRC) -o $(OBJ)