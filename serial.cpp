#include <stdio.h>

int main(int argc, char const *argv[])
{
	FILE *fp;
	fp=fopen("serial.dat", "a");
	fprintf(fp, "1\t23\n");
    /* CODE */

    fclose(fp);
	return 0;
}