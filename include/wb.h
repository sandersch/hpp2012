#ifndef _WB_H
#define _WB_H 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef NOGPU
#define __global__
void cudaThreadSynchronize() {}
#else
#include <cuda.h>
#endif

typedef struct {
        int argc;
        char **argv;
} wbArg_t;


typedef enum {
        Generic,
        GPU,
        Compute,
        Copy
} stage_t;


typedef enum {
        TRACE
} logLevel_t;

char* logNames[] = { "TRACE"};

wbArg_t wbArg_read(int argc, char **argv)
{
        wbArg_t args;
        args.argc = argc;
        args.argv = argv;
        return args;
}

FILE* wbArg_getInputFile(wbArg_t args, int index)
{
        FILE* input = fopen(args.argv[index+1], "r");
        if (input == NULL)
        {
                fprintf(stderr, "Couldn't open %s\n", args.argv[index+1]);
        }

        return input;
}


/*
Assumes input is a number per line
 */
float* wbImport(FILE* input, int* length)
{
        int currentBufferSize = 0;
        int counter = 0;
        int incrementSize = 256;
        float* contents;
        char readBuffer[100];
        float number;

        *length = 0;
        contents = NULL;
        while (fgets(readBuffer, 100, input))
        {
                /* make sure we read the whole line */
                char* newline = strchr(readBuffer, '\n');
                if (!newline)
                {
                        fprintf(stderr, "Line %d is too long\n", counter);
                        fprintf(stderr, "%s\n", readBuffer);
                        fprintf(stderr, "%d\n", strlen(readBuffer));
                        return contents;
                }
                else if (newline == readBuffer)
                {
                        /* Empty line */
                        continue;
                }

                *newline = '\0';

                counter++;
                if (counter > currentBufferSize)
                {
                        /* need to expand buffer */
                        currentBufferSize += incrementSize;
                        contents = (float *) realloc((void *) contents, sizeof(float) * currentBufferSize);
                }

                /* convert string to float */
                number = atof(readBuffer);

                contents[counter-1] = number;

        }
        *length = counter;

        return contents;
}

void wbTime_start(stage_t stage, char* message)
{

}

void wbTime_stop(stage_t stage, char* message)
{

}

void wbLog(logLevel_t logLevel, char* message, int number)
{
        fprintf(stderr, "%s: %s%d\n", logNames[logLevel], message, number);
}

void wbSolution(wbArg_t args, float* output, int length)
{

}

#endif /* _WB_H */
