#include "arguproc.h"
/* check argument */
void Argumentprocess(int argc, char **argv, ARGUTYPE *parg)
{
    int opt;
    char *inputFile;
    char *outputFile;

    if (argc <= 1)
    {
        usage();
        exit(-1);
    }

    while ((opt = getopt(argc, argv, "I:O:")) != -1)
    {
        switch (opt)
        {
        case 'I':
            inputFile = optarg;
            break;

        case 'O':
            outputFile = optarg;
            break;

        default:
            usage();
            exit(-1);
        }
    }

    // Check if inputFile and outputFile are set
    if (inputFile == NULL || outputFile == NULL)
    {
        fprintf(stderr, "Input and output files must be specified.\n");
        usage();
        exit(EXIT_FAILURE);
    }

    // process input file
    FILE *file_in = fopen(inputFile, "r");
    if (file_in == NULL)
    {
        perror("Failed to open the file");
        exit(EXIT_FAILURE);
    }

    char line_in[256]; // store each line of the file

    int flag = 0;
    for (int i = 1; i <= 9 && fgets(line_in, sizeof(line_in), file_in) != NULL; i++)
    {
        line_in[strcspn(line_in, "\n")] = '\0';
        char *file_content_in = strdup(line_in);
        if (file_content_in == NULL)
        {
            perror("Failed to allocate memory for input content");
            fclose(file_in);
            exit(EXIT_FAILURE);
        }

        // set the corresponding input file name
        switch (i)
        {
        case 1:
            parg->ee_in = file_content_in;
            flag++;
            break;
        case 2:
            parg->en_in = file_content_in;
            flag++;
            break;
        case 3:
            parg->ez_in = file_content_in;
            flag++;
            break;
        case 4:
            parg->ne_in = file_content_in;
            flag++;
            break;
        case 5:
            parg->nn_in = file_content_in;
            flag++;
            break;
        case 6:
            parg->nz_in = file_content_in;
            flag++;
            break;
        case 7:
            parg->ze_in = file_content_in;
            flag++;
            break;
        case 8:
            parg->zn_in = file_content_in;
            flag++;
            break;
        case 9:
            parg->zz_in = file_content_in;
            flag++;
            break;
        }
    }

    fclose(file_in);

    // Check if exactly 9 lines were processed in input file
    if (flag < 9)
    {
        fprintf(stderr, "Insufficient lines in input file.\n");
        exit(EXIT_FAILURE);
    }

    // process output file
    FILE *file_out = fopen(outputFile, "r");
    if (file_out == NULL)
    {
        perror("Failed to open the file");
        exit(EXIT_FAILURE);
    }

    flag = 0;
    char line_out[256]; // store each line of the file
    for (int i = 1; i <= 9 && fgets(line_out, sizeof(line_out), file_out) != NULL; i++)
    {
        line_out[strcspn(line_out, "\n")] = '\0';

        char *file_content_out = strdup(line_out);
        if (file_content_out == NULL)
        {
            perror("Failed to allocate memory for output content");
            fclose(file_out);
            exit(EXIT_FAILURE);
        }

        switch (i)
        {
        case 1:
            parg->rr_out = file_content_out;
            flag++;
            break;
        case 2:
            parg->rt_out = file_content_out;
            flag++;
            break;
        case 3:
            parg->rz_out = file_content_out;
            flag++;
            break;
        case 4:
            parg->tr_out = file_content_out;
            flag++;
            break;
        case 5:
            parg->tt_out = file_content_out;
            flag++;
            break;
        case 6:
            parg->tz_out = file_content_out;
            flag++;
            break;
        case 7:
            parg->zr_out = file_content_out;
            flag++;
            break;
        case 8:
            parg->zt_out = file_content_out;
            flag++;
            break;
        case 9:
            parg->zz_out = file_content_out;
            flag++;
            break;
        }
    }

    fclose(file_out);

    // Check if exactly 9 lines were processed in output file
    if (flag < 9)
    {
        fprintf(stderr, "Insufficient lines in output file.\n");
        exit(EXIT_FAILURE);
    }
}
/* end of parsing command line arguments */