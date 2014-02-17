/*
 * cpuknapsack.c
 *
 *      Author: Enrico Zamagni
 */

#include "genetic.h"

int isBetterSolution(SolutionVal *i1, SolutionVal *i2) {
	return i1->fitness > i2->fitness
			|| (i1->fitness == i2->fitness && i1->weight < i2->weight);
}

void generation_run(Individual *pop, Individual *child, KPInstance inst, Params param) {
	int r, c, i, w, id;
	dim3 coord = { 0, 0, 0 };

	// create individual stub
	Bitset bstub = createBitset(inst.numvar);
	free(bstub.bits);

	for (i = 0; i < param.iteration_per_run; i++) {
		coord.x = 0;
		coord.y = 0;
		// compute run for every individual in population matrix
		for (r = 0; r < param.matrixSize.y; r++) {
			for (c = 0; c < param.matrixSize.x; c++) {
				id = r * param.matrixSize.y + c;
				evaluateIndividual(&pop[id], inst, param.penalratio);

				// generate new Individual
				for(w = 0; w < bstub.wordCount; w++)
					child[id].bitset.bits[w] = random() & 0xffffffff;

				evaluateIndividual(&child[id], inst, param.penalratio);

				if (isBetterSolution(&child[id].value, &pop[id].value))
					copyIndividual(&child[id], &pop[id]);

				coord.x++;
			}
			coord.x = 0;
			coord.y++;
		}
	}
}

int main(int argc, char* argv[]) {
	if (argc <= 1) {
		printHelp();
		exit(0);
	}

	// load problem instance
	KPInstance inst = loadInstance(argv[1]);

	// create individual stub
	Bitset stubbs = createBitset(inst.numvar);
	free(stubbs.bits);

	// setup application & parameters
	Params params = setupParams(argc, argv);
	int popCount = params.matrixSize.x * params.matrixSize.y;

	printf("Using seed: %lu\n", params.seed);
	if (params.stepmode) {
		params.iteration_per_run = 1;
		printf("[WARNING] Using step-by-step verbose mode. "
				"Press ENTER to proceed through iterations.\n");
	}

	printf("Population matrix is %d x %d = %d individuals\n",
			params.matrixSize.x, params.matrixSize.y, popCount);
	printf("Setting up initial population...");

	// instantiate population
	int i;
	Individual *pop, *childpop;
	pop = (Individual*) malloc(popCount * sizeof(Individual));
	childpop = (Individual*) malloc(popCount * sizeof(Individual));
	for (i = 0; i < popCount; i++) {
		createIndividual(&pop[i], stubbs.bitCount);
		createEmptyIndividual(&childpop[i], stubbs.bitCount);
	}

	srandom(params.seed);
	printf(" done.\nNow searching:\n");

	char stroutput[MAXSTRLEN];
	int iterCount = 0, forcequit = 0;
	SolutionVal bestVal = { 0, 0 };

	do {
		if (params.stepmode) {
			char userin = '\0';
			popdump(pop, params.matrixSize, stroutput);
			while (userin != '\n' && userin != 'q')
				userin = getchar();
			if (userin == 'q')
				forcequit = TRUE;
		}

		generation_run(pop, childpop, inst, params);
		iterCount += params.iteration_per_run;

		// check for best solution
		bestVal.fitness = 0;
		bestVal.weight = 0;
		for (i = 0; i < popCount; i++) {
			if (isBetterSolution(&pop[i].value, &bestVal)) {
				stubbs.bits = pop[i].bitset.bits;
				bestVal = pop[i].value;
			}
		}

		if (params.verbose) {
			// display best solution
			printf("%4d random iterations performed. ", iterCount);
			if(params.verbose == 1) {
				printBitset(&stubbs, stroutput);
				printf("Best solution so far: %s with fitness=%d and weight=%d\n",
					stroutput, bestVal.fitness, bestVal.weight);
			} else {
				printf("Best solution so far has fitness=%d and weight=%d\n",
					bestVal.fitness, bestVal.weight);
			}
		}
	} while (!forcequit && !satcondition(params, iterCount, bestVal, inst));

	// print solution, if found
	printf("Search terminated.\n");
	if (!params.verbose) {
		printf("%4d random iterations performed.\n", iterCount);
		printBitset(&stubbs, stroutput);
	}
	if (bestVal.weight <= inst.capacity && bestVal.fitness > 0) {
		printf("Best found solution: %s\nFitness = %d\nWeight = %d out of %d (%.1f%%)\n",
			stroutput, bestVal.fitness, bestVal.weight, inst.capacity, ((float)(bestVal.weight * 100)) / inst.capacity);
	} else {
		printf("No feasible solution found.\n");
	}

	// cleanup
	free(pop);
	free(childpop);
	destroyInstance(&inst);

	return 0;
}
