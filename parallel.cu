#include <stdio.h>
#include "cuda.h"

int main(int argc, char const *argv[])
{
	FILE *fp;
    fp=fopen("parallel.dat", "w");
    fprintf(fp, "# NumberOfElements\ttime\n");
    fprintf(fp, "1\t20\n");
    fprintf(fp, "2\t20\n");
    fprintf(fp, "3\t20\n");
    /* CODE */

    fclose(fp);
	return 0;
}