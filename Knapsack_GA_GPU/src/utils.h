/*
 * utils.h
 *
 *      Author: Enrico Zamagni
 */

/*
 *  Fix for Eclipse syntax error checking
 */
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#define __constant__
#define __host__
#endif

#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include "bitset.h"

#define SEARCHMODE_RUNLIMIT 0
#define SEARCHMODE_FIRSTFEASIBLE 1

#define MAXSTRLEN 1024
#define FALSE 0
#define TRUE 1
#define DEFAULT_NUMBLOCKS 1
#define DEFAULT_DEVICE 0
#define DEFAULT_MSQRSIZE 10
#define DEFAULT_POPSIZE DEFAULT_MSQRSIZE * DEFAULT_MSQRSIZE
#define DEFAULT_PROB_MUTATION .1f
#define DEFAULT_PROB_CROSSOVER .80f
#define DEFAULT_MAX_ITERATIONS 500
#define DEFAULT_ITERATIONS_PER_RUN 100
//#define DEFAULT_PROB_RADIOACTIVITY 0
#define DEFAULT_PENALIZATION_RATIO 2.0f
#define CHAR_MUTATION 'm'
#define CHAR_CROSSOVER 'c'
#define CHAR_SEED 's'
#define CHAR_POPSIZE 'p'
#define CHAR_SETDEV 'd'
#define CHAR_ITERATIONS 'r'
#define CHAR_BLOCKS 'i'
#define CHAR_VERBOSE 'v'
#define CHAR_STEPMODE 'x'
#define CHAR_HEXPRINT 'h'
#define CHAR_SEARCHMODE 'o'
#define CHAR_BUILDMODE 'b'
#define CHAR_PENALIZE 'z'

#define CUDA_CHECK_ERROR() \
  {\
    cudaError_t ce = cudaGetLastError();\
    if(ce != cudaSuccess) { \
      printf("[ERROR] %s\n", cudaGetErrorString(ce));\
      exit(EXIT_FAILURE);\
    }\
  }

typedef unsigned int uint;

typedef struct {
	int fitness;
	int weight;
} SolutionVal;

// knapsack problem instance
typedef struct {
	uint capacity;
	uint numvar;
	uint *weight;
	uint *profit;
} KPInstance;

// application parameters
typedef struct {
	unsigned long seed;
	dim3 matrixSize;
	int nblocks;
	int device;
	int iteration_limit;
	int iteration_per_run;
	float p_mutation;
	float p_crossover;
	int verbose;
	int stepmode;
	int searchmode;
	int buildmode;
	float penalratio;
} Params;


void destroyInstance(KPInstance *inst) {
	free(inst->profit);
	free(inst->weight);
}

KPInstance loadInstance(char fileName[]) {
	int i, nvar;
	FILE *inf;
	KPInstance inst;

	inf = fopen(fileName, "r");
	if (inf == NULL) {
		printf("ERROR: could not open instance file!\n");
		exit(1);
	}

	// read num. of objects
	fscanf(inf, "%d", &nvar);
	if (nvar <= 1) {
		printf("ERROR: invalid number of objects in problem instance!\n");
		exit(1);
	}

	inst.numvar = nvar;
	inst.profit = (uint*) malloc(nvar * sizeof(uint));
	inst.weight = (uint*) malloc(nvar * sizeof(uint));

	// read object profit\weight
	for (i = 0; i < nvar; i++) {
		if (fscanf(inf, " %*d %d %d", &inst.profit[i], &inst.weight[i]) == EOF) {
			printf("ERROR: unexpected end of file!\n");
			exit(1);
		}
	}

	// read knapsack capacity
	if (fscanf(inf, "%d", &inst.capacity) == EOF || inst.capacity <= 0) {
		printf("ERROR: invalid capacity in problem instance!\n");
		exit(1);
	}

	fclose(inf);
	return inst;
}

dim3 fitMatrixSize(int popCount) {
	dim3 result;
	result.y = floor(sqrt((float)popCount));
	result.x = popCount / result.y;
	return result;
}

Params setupParams(int argc, char* argv[]) {
	Params params;
	uint popSize;

	// set defaults
	popSize = DEFAULT_POPSIZE;
	params.device = DEFAULT_DEVICE;
	params.seed = 0;
	params.p_crossover = DEFAULT_PROB_CROSSOVER;
	params.p_mutation = DEFAULT_PROB_MUTATION;
	params.iteration_limit = DEFAULT_MAX_ITERATIONS;
	params.iteration_per_run = DEFAULT_ITERATIONS_PER_RUN;
	params.verbose = FALSE;
	params.stepmode = FALSE;
	params.buildmode = FALSE;
	params.nblocks = DEFAULT_NUMBLOCKS;
	params.searchmode = SEARCHMODE_RUNLIMIT;
	params.penalratio = DEFAULT_PENALIZATION_RATIO;

	// parse user params
	while (argc > 1) {
		argv++;
		argc--;
		if (argv[0][0] != '-')
			continue;

		switch (argv[0][1]) {

		case CHAR_BUILDMODE: //build mode
			params.buildmode = TRUE;
			break;

		case CHAR_VERBOSE: // verbose
			params.verbose = TRUE;
			if(argv[0][2] == '0') params.verbose++;
			break;

		case CHAR_HEXPRINT: //hex printing
			hexprint = TRUE;
			break;

		case CHAR_STEPMODE: //step by step mode
			params.stepmode = TRUE;
			break;

		case CHAR_SEARCHMODE: //search mode
			params.searchmode = atoi(argv[1]);
			if(params.searchmode < 0 || params.searchmode > 1) {
				printf("Unknown search mode entered: keeping default value.\n");
				params.searchmode = SEARCHMODE_RUNLIMIT;
			}
			argv++;
			argc--;
			break;

		case CHAR_SEED: // seed for random numbers
			params.seed = strtoul(argv[1], (char**) NULL, 10);
			argv++;
			argc--;
			break;

		case CHAR_POPSIZE: // population size
			popSize = atoi(argv[1]);
			if (popSize <= 1) {
				printf("Wrong number of population size entered: keeping default value.\n");
				popSize = DEFAULT_POPSIZE;
			}
			argv++;
			argc--;
			break;

		case CHAR_SETDEV: // cuda device
			params.device = atoi(argv[1]);
			if (params.device < 0) {
				printf("Wrong device number entered: keeping default.\n");
				params.device = DEFAULT_DEVICE;
			}
			argv++;
			argc--;
			break;

		case CHAR_CROSSOVER: //crossover probability
			params.p_crossover = (float) atof(argv[1]);
			if (params.p_crossover < 0 || params.p_crossover > 1) {
				printf("Probability must be contained between 0 and 1: keeping default value.\n");
				params.p_crossover = DEFAULT_PROB_CROSSOVER;
			}
			argv++;
			argc--;
			break;
		case CHAR_MUTATION: //mutation probability
			params.p_mutation = (float) atof(argv[1]);
			if (params.p_mutation < 0 || params.p_mutation > 1) {
				printf("Probability must be contained between 0 and 1: keeping default value.\n");
				params.p_mutation = DEFAULT_PROB_MUTATION;
			}
			argv++;
			argc--;
			break;
		case CHAR_ITERATIONS: //iteration limit
			params.iteration_limit = atoi(argv[1]);
			if(params.iteration_limit <= 0) {
				printf("Wrong iteration limit entered: keeping default value.\n");
				params.iteration_limit = DEFAULT_MAX_ITERATIONS;
			}
			argv++;
			argc--;
			break;
		case CHAR_BLOCKS: //island number
			params.nblocks = atoi(argv[1]);
			if(params.nblocks <= 0) {
				printf("Wrong number of islands entered: keeping default value.\n");
				params.nblocks = DEFAULT_NUMBLOCKS;
			}
			argv++;
			argc--;
			break;
		case CHAR_PENALIZE: //penalization ratio
			if(argv[0][2] == CHAR_PENALIZE) {
				params.penalratio = HUGE;
			} else {
				params.penalratio = (float) atof(argv[1]);
				if (params.penalratio < 0) {
					printf("Wrong penalization ratio entered: keeping default value.\n");
					params.penalratio = DEFAULT_PENALIZATION_RATIO;
				}
				argv++;
				argc--;
			}
			break;
		}
	}

	params.matrixSize = fitMatrixSize(popSize);
	if (params.seed == 0) {
		// generate random seed
		params.seed = time(NULL);
	}

	return params;
}

void printHelp() {
	printf("Usage: gpuknapsack filename [options]\n");
	printf("Where options are:\n"
			"\t-%c sets the random seed\n"
			"\t-%c sets the population size\n"
			"\t-%c sets the number of islands\n"
			"\t-%c sets the CUDA device\n"
			"\t-%c sets the crossover probability\n"
			"\t-%c sets the mutation probability\n"
			"\t-%c sets the iteration threshold\n"
			"\t-%c enables step-by-step mode\n"
			"\t-%c enables verbose mode\n"
			"\t-%c enables build mode\n"
			"\t-%c prints bitsets in hexadecimal format\n"
			"\t-%c sets the penalization ratio\n", CHAR_SEED, CHAR_POPSIZE, CHAR_BLOCKS,
			CHAR_SETDEV, CHAR_CROSSOVER, CHAR_MUTATION, CHAR_ITERATIONS, CHAR_STEPMODE, CHAR_VERBOSE,
			CHAR_BUILDMODE, CHAR_HEXPRINT, CHAR_PENALIZE);
}

int satcondition(Params par, int iterCount, SolutionVal bestSol, KPInstance inst) {

	switch (par.searchmode) {
		case SEARCHMODE_RUNLIMIT:
			return iterCount >= par.iteration_limit;
		case SEARCHMODE_FIRSTFEASIBLE:
			return bestSol.weight <= inst.capacity && bestSol.fitness > 0;
	}
	return FALSE;
}

__host__
__device__
int isBetterSolution(SolutionVal *i1, SolutionVal *i2) {
	return i1->fitness > i2->fitness
			|| (i1->fitness == i2->fitness && i1->weight < i2->weight);
}

///////////////////////////////////////////////////////////////////
//////////////          DUMP FUNCTIONS           //////////////////
///////////////////////////////////////////////////////////////////

void memwdump(word *addr, uint wcount) {

	printf("------------------------\n");
	int i;
	for (i = 0; i < wcount; i++) {
		printf("%08X ", addr[i]);
	}
	printf("\n---- %d word dumped ----\n\n", i);
}

void memdump(unsigned char *addr, uint bcount) {

	printf("------------------------\n");
	int i;
	for (i = 0; i < bcount; i++) {
		printf("%02X ", addr[i]);
	}
	printf("\n---- %d bytes dumped ----\n\n", i);
}

void popdump(word *popaddr, Bitset *stubbs, dim3 popsize, char* stroutput) {
	int x, y;
	for(y = 0; y < popsize.y; y++) {
		for(x = 0; x < popsize.x; x++) {
			stubbs->bits = &popaddr[(y * popsize.y + x) * stubbs->wordCount];
			printBitset(stubbs, stroutput);
			printf("%s ", stroutput);
		}
		printf("\n");
	}
	printf("\n");
}

///////////////////////////////////////////////////////////////////
/////////////          DEVICE FUNCTIONS           /////////////////
///////////////////////////////////////////////////////////////////

__device__
int nextRandomInt(curandState *randState, int minVal, int maxVal) {
	unsigned int rnd = curand(randState);
	return minVal + rnd % (maxVal - minVal + 1);
}

__device__
int getNeighbourId(int dir) {
	dim3 coord = threadIdx;
	int tid;

	// calculate desired neighbour coordinates
	switch (dir) {
	case 0: //UP
		if (coord.y == 0)
			coord.y = blockDim.y - 1;
		else
			coord.y--;
		break;
	case 1: //RIGHT
		if (coord.x == blockDim.x - 1)
			coord.x = 0;
		else
			coord.x++;
		break;
	case 2: //DOWN
		if (coord.y == blockDim.y - 1)
			coord.y = 0;
		else
			coord.y++;
		break;
	case 3: //LEFT
		if (coord.x == 0)
			coord.x = blockDim.x - 1;
		else
			coord.x--;
		break;
	}

	tid = coord.y * blockDim.x + coord.x;
	return (blockDim.x * blockDim.y) * blockIdx.x + tid;
}

__device__
int getNeighbourIdx(int dir) {
	dim3 nc = threadIdx;

	// calculate desired neighbour coordinates
	switch (dir) {
	case 0: //UP
		if (nc.y == 0)
			nc.y = blockDim.y - 1;
		else
			nc.y--;
		break;
	case 1: //RIGHT
		if (nc.x == blockDim.x - 1)
			nc.x = 0;
		else
			nc.x++;
		break;
	case 2: //DOWN
		if (nc.y == blockDim.y - 1)
			nc.y = 0;
		else
			nc.y++;
		break;
	case 3: //LEFT
		if (nc.x == 0)
			nc.x = blockDim.x - 1;
		else
			nc.x--;
		break;
	}
	return nc.y * blockDim.x + nc.x;
}
