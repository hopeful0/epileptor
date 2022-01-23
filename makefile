COMPILER=gcc -std=c99 -O3

epileptor: csrc/epileptor.c
	${COMPILER} csrc/epileptor.c -lm -fopenmp -o epileptor

lib: csrc/epileptor.c
	${COMPILER} csrc/epileptor.c -lm -fopenmp -fPIC -shared -o epileptor.so


clean:
	rm -rf *.o