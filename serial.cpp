#include <stdio.h>

int main(int argc, char const *argv[])
{
	FILE *fp;
	fp=fopen("serial.dat", "w");
    fprintf(fp, "# NumberOfElements\ttime\n");
	fprintf(fp, "1\t23\n");
    fprintf(fp, "2\t23\n");
    fprintf(fp, "3\t23\n");
    /* CODE */

    fclose(fp);
	return 0;
}