/*
 * utils.h
 *
 *      Author: Enrico Zamagni
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <math.h>
#include <time.h>
#include "bitset.h"

#define SEARCHMODE_RUNLIMIT 0
#define SEARCHMODE_FIRSTFEASIBLE 1

#define MAXSTRLEN 1024
#define FALSE 0
#define TRUE 1
#define DEFAULT_MSQRSIZE 10
#define DEFAULT_POPSIZE DEFAULT_MSQRSIZE * DEFAULT_MSQRSIZE
#define DEFAULT_PROB_MUTATION .1f
#define DEFAULT_PROB_CROSSOVER .80f
#define DEFAULT_MAX_ITERATIONS 500
#define DEFAULT_ITERATIONS_PER_RUN 100
#define DEFAULT_PENALIZATION_RATIO 2.0f
#define CHAR_MUTATION 'm'
#define CHAR_CROSSOVER 'c'
#define CHAR_SEED 's'
#define CHAR_POPSIZE 'p'
#define CHAR_ITERATIONS 'r'
#define CHAR_VERBOSE 'v'
#define CHAR_STEPMODE 'x'
#define CHAR_HEXPRINT 'h'
#define CHAR_SEARCHMODE 'o'
#define CHAR_BUILDMODE 'b'
#define CHAR_PENALIZE 'z'

typedef struct {
	int x;
	int y;
	int z;
} dim3;

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
	if(inf == NULL) {
		printf("ERROR: could not open instance file!\n");
		exit(1);
	}

	// read num. of objects
	fscanf(inf, "%d", &nvar);
	if(nvar <= 1) {
		printf("ERROR: invalid number of objects in problem instance!\n");
		exit(1);
	}

	inst.numvar = nvar;
	inst.profit = (uint*)malloc(nvar * sizeof(uint));
	inst.weight = (uint*)malloc(nvar * sizeof(uint));

	// read object profit\weight
	for(i = 0; i < nvar; i++) {
		if(fscanf(inf, " %*d %d %d", &inst.profit[i], &inst.weight[i]) == EOF) {
			printf("ERROR: unexpected end of file!\n");
			exit(1);
		}
	}

	// read knapsack capacity
	if(fscanf(inf, "%d", &inst.capacity) == EOF || inst.capacity <= 0) {
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
	params.seed = 0;
	params.p_crossover = DEFAULT_PROB_CROSSOVER;
	params.p_mutation = DEFAULT_PROB_MUTATION;
	params.iteration_limit = DEFAULT_MAX_ITERATIONS;
	params.iteration_per_run = DEFAULT_ITERATIONS_PER_RUN;
	params.verbose = FALSE;
	params.stepmode = FALSE;
	params.buildmode = FALSE;
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
			"\t-%c sets the crossover probability\n"
			"\t-%c sets the mutation probability\n"
			"\t-%c sets the iteration threshold\n"
			"\t-%c enables step-by-step mode\n"
			"\t-%c enables verbose mode\n"
			"\t-%c enables build mode\n"
			"\t-%c prints bitsets in hexadecimal format\n"
			"\t-%c sets the penalization ratio\n", CHAR_SEED, CHAR_POPSIZE,
			CHAR_CROSSOVER, CHAR_MUTATION, CHAR_ITERATIONS, CHAR_STEPMODE,
			CHAR_VERBOSE, CHAR_BUILDMODE, CHAR_HEXPRINT, CHAR_PENALIZE);
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

///////////////////////////////////////////////////////////////////
/////////////          UTILITY FUNCTIONS           ////////////////
///////////////////////////////////////////////////////////////////

float nextRandomFloat() {
	return (float)rand() / (float)RAND_MAX;
}

int nextRandomInt(int minVal, int maxVal) {
	return minVal + random() % (maxVal - minVal + 1);
}

int getNeighbourId(dim3 coord, dim3 matrixDim, int dir) {
	// calculate desired neighbour coordinates
	switch (dir) {
	case 0: //UP
		if (coord.y == 0)
			coord.y = matrixDim.y - 1;
		else
			coord.y--;
		break;
	case 1: //RIGHT
		if (coord.x == matrixDim.x - 1)
			coord.x = 0;
		else
			coord.x++;
		break;
	case 2: //DOWN
		if (coord.y == matrixDim.y - 1)
			coord.y = 0;
		else
			coord.y++;
		break;
	case 3: //LEFT
		if (coord.x == 0)
			coord.x = matrixDim.x - 1;
		else
			coord.x--;
		break;
	}

	return coord.y * matrixDim.x + coord.x;
}

#endif /* UTILS_H_ */
