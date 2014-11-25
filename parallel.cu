#include <stdio.h>
#include "cuda.h"

int main(int argc, char const *argv[])
{
	FILE *fp;
    fp=fopen("parallel.dat", "a");
    fprintf(fp, "1\t20\n");
    /* CODE */

    fclose(fp);
	return 0;
}