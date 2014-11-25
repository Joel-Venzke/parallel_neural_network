#include <stdio.h>
#include "cuda.h"

int main(int argc, char const *argv[])
{
	FILE *fp;
    fp=fopen("data/parallel.dat", "a");
    fprintf(fp, "1\t20\n");
    clock_t t;
	t = clock();

	/* code */

    t = clock() - t;
    fprintf (fp, "%d\t%f\n", SIZE,((float)t)/CLOCKS_PER_SEC);
    fclose(fp);
	return 0;
}