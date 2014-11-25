#include <stdio.h>
#include <time.h>
#define SIZE 10000

int main(int argc, char const *argv[])
{
	FILE *fp;
	fp=fopen("data/serial.dat", "a");
	clock_t t;
	t = clock();

	/* code */

    t = clock() - t;
    fprintf (fp, "%d\t%f\n", SIZE,((float)t)/CLOCKS_PER_SEC);
    fclose(fp);
	return 0;
}