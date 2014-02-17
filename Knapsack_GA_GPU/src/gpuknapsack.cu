
/*
 * Fix for Eclipse syntax error checking
 */
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#define __constant__
#define blockIdx threadIdx
#define CUDA_KERNEL_LAUNCH(...)
#else
#define CUDA_KERNEL_LAUNCH(...)  <<< __VA_ARGS__ >>>
#endif

#include <cuda_runtime_api.h>
#include "utils.h"

// addressing functions for cuda kernels
#define IDX(ui) ui * bstub.wordCount

__device__ __constant__ Bitset bstub; 		// generic bitset stub
__device__ __constant__ KPInstance pinst; 	// problem instance
__device__ __constant__ float mutationp;	// mutation probability
__device__ __constant__ float crossoverp;	// crossover probability
__device__ __constant__ float penalratio;	// penalization ratio
__device__ __constant__ int run_limit;		// thread run limit


__device__
SolutionVal evaluateSubject(word *subject) {
	SolutionVal result = {0, 0};
	word w, mask = 1u << (WORDLEN - 1);
	int b, curWord = 0;

	// process every bit
	for (b = 0; b < bstub.bitCount; b++) {
		if (b % WORDLEN == 0) {
			w = subject[curWord];
			curWord++;
		}
		if (w & mask) {
			result.fitness += pinst.profit[b];
			result.weight += pinst.weight[b];
		}
		w <<= 1;
	}

	// penalize fitness if unfeasible
	if(result.weight > pinst.capacity) {
		int overweight = result.weight - pinst.capacity;
		result.fitness -= overweight * penalratio;
	}

	return result;
}

__device__
void performCrossoverOnePoint(word *parent, word *partner,
		word *child, int pcut, int direction) {
	int bitCopied = 0;

	// create mask
	word mask = 0xffffffff << (WORDLEN - (pcut % WORDLEN));

	// transfer genes to offspring..
	while (bitCopied < bstub.bitCount) {
		if (bitCopied + WORDLEN < pcut) {
			// ..before the cut
			direction? *child = *parent : *child = *partner;
		} else if (bitCopied > pcut) {
			//.. after the cut
			direction? *child = *partner: *child = *parent;
		} else {
			// ..into the cut
			direction? *child = (*parent & mask) | (*partner & ~mask)
					: *child = (*partner & mask) | (*parent & ~mask);
		}
		parent++; partner++;
		child++;
		bitCopied += WORDLEN;
	}
}

__device__
void performMutation(word *subject, int pos) {
	int selWord = pos / WORDLEN;
	word mask = 1u << (WORDLEN - pos % WORDLEN - 1);
	subject[selWord] ^= mask;
}

__device__
void bsCopy(word *source, word *dest) {
	int w;
	for(w = 0; w < bstub.wordCount; w++)
		dest[w] = source[w];
}


__global__
void generation_run(word *cPool, word *childPool, word *tempPool, SolutionVal *valPool, curandState *randState) {
	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int utid = (blockDim.x * blockDim.y) * blockIdx.x + tid;

	// fetch memory
	int i, n, t, dir;
	char adopt;

	// setup individual
	SolutionVal myval, bestval, childval;
	myval = evaluateSubject(&cPool[IDX(utid)]);

	// setup random state
	curandState localState = randState[utid];
	// compute run
	for(i = 0; i < run_limit; i++) {
		bestval = myval;
		adopt = FALSE;

		// recombination
		if(curand_uniform(&localState) < crossoverp) {
			// for each neighbour
			for(n = 0; n < 4; n++) {
				// get a random pointcut..
				t = nextRandomInt(&localState, 1, bstub.bitCount - 1);
				// perform two crossovers
				for(dir = 0; dir < 2; dir++) {
					performCrossoverOnePoint(&cPool[IDX(utid)],
							&cPool[IDX(getNeighbourId(n))],
							&childPool[IDX(utid)], t, dir);

					if(curand_uniform(&localState) < mutationp) {
						// mutation passed
						t = nextRandomInt(&localState, 0, bstub.bitCount - 1);
						performMutation(&childPool[IDX(utid)], t);
					}

					// store child if he's a better solution
					childval = evaluateSubject(&childPool[IDX(utid)]);
					if(childval.fitness > bestval.fitness
							|| (childval.fitness == bestval.fitness && childval.weight < bestval.weight)) {
						adopt = TRUE;
						bestval = childval;
						bsCopy(&childPool[IDX(utid)], &tempPool[IDX(utid)]);
					}
				}
			}
		}

		// wait util all threads completed this iteration
		__syncthreads();
		// eventually adopt child
		if(adopt) {
			myval = bestval;
			bsCopy(&tempPool[IDX(utid)], &cPool[IDX(utid)]);
		}
		__syncthreads();
	}

	// return
	randState[utid] = localState;
	valPool[utid] = myval;
}

__global__
void setup(word *cGlobalPool, curandState *randState, unsigned long long seed, int buildmode) {
	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int utid = (blockDim.x * blockDim.y) * blockIdx.x + tid;

	// setup cuda random states
	curandState localState;
	curand_init(seed, utid, 0, &localState);

	// setup initial population
	int i;
	for (i = 0; i < bstub.wordCount; i++) {
		if(buildmode) {
			cGlobalPool[IDX(utid) + i] = 0;
		} else {
			cGlobalPool[IDX(utid) + i] = curand(&localState);
		}
	}

	// save cuda random state
	randState[utid] = localState;
}


int main(int argc, char* argv[]) {
	if(argc <= 1) {
		printHelp();
		exit(0);
	}

	// load problem instance
	KPInstance inst = loadInstance(argv[1]);
	KPInstance d_inst = inst;

	// create individual stub
	Bitset stubbs = createBitset(inst.numvar);

	// setup application & parameters
	Params params = setupParams(argc, argv);
	int popCount = params.matrixSize.x * params.matrixSize.y;
	int gpoolsize, valpoolsize;
	gpoolsize = popCount * stubbs.wordCount * sizeof(word) * params.nblocks;
	valpoolsize = popCount * params.nblocks * sizeof(SolutionVal);
	if(params.device != DEFAULT_DEVICE) {
		cudaSetDevice(params.device);
		printf("Using device %d\n", params.device);
	}
	printf("Using seed: %lu\n", params.seed);
	if(params.stepmode) {
		params.iteration_per_run = 1;
		printf("[WARNING] Using step-by-step verbose mode. "
				"Press ENTER to proceed through iterations.\n");
	}

	printf("Population matrix is %d x %d = %d individuals\n", params.matrixSize.x, params.matrixSize.y, popCount);
	printf("Setting up initial population...");

	// transfer instance parameters & constants to device
	uint *d_profit, *d_weight;
	cudaMalloc(&d_profit, inst.numvar * sizeof(uint));
	cudaMalloc(&d_weight, inst.numvar * sizeof(uint));
	cudaMemcpy(d_profit, inst.profit, inst.numvar * sizeof(uint), cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight, inst.weight, inst.numvar * sizeof(uint), cudaMemcpyHostToDevice);
	d_inst.profit = d_profit; d_inst.weight = d_weight;
	cudaMemcpyToSymbol(pinst, &d_inst, sizeof(KPInstance));
	cudaMemcpyToSymbol(bstub, &stubbs, sizeof(Bitset));
	cudaMemcpyToSymbol(mutationp, &params.p_mutation, sizeof(params.p_mutation));
	cudaMemcpyToSymbol(crossoverp, &params.p_crossover, sizeof(params.p_crossover));
	cudaMemcpyToSymbol(penalratio, &params.penalratio, sizeof(params.penalratio));
	cudaMemcpyToSymbol(run_limit, &params.iteration_per_run, sizeof(int));

	// instantiate device memory
	word *d_gpool, *h_gpool, *d_childpool, *d_temppool;
	SolutionVal *d_valpool, *h_valpool;
	curandState *d_States;
	h_gpool = (word*) malloc(gpoolsize);
	h_valpool = (SolutionVal*) malloc(valpoolsize);
	cudaMalloc(&d_gpool, gpoolsize);
	cudaMalloc(&d_childpool, gpoolsize);
	cudaMalloc(&d_temppool, gpoolsize);
	cudaMalloc(&d_valpool, valpoolsize);
	cudaMalloc(&d_States, params.nblocks * popCount * sizeof(curandState));
	CUDA_CHECK_ERROR();

	// setup initial population
	setup CUDA_KERNEL_LAUNCH(params.nblocks, params.matrixSize)(d_gpool, d_States, params.seed, params.buildmode);
	printf(" (using %d bytes (%d KB) of device memory)", gpoolsize * 3, (int)ceil((float)(gpoolsize * 3) / 1024));
	CUDA_CHECK_ERROR();
	cudaDeviceSynchronize();
	printf(" done.\nNow searching:\n");

	char stroutput[MAXSTRLEN];
	int iterCount = 0, i, forcequit = 0;
	SolutionVal bestVal = {0, 0};
	stubbs.bits = h_gpool;

	if(params.stepmode) cudaMemcpy(h_gpool, d_gpool, gpoolsize, cudaMemcpyDeviceToHost);

	do {
		if(params.stepmode) {
			char userin = '\0';
			popdump(h_gpool, &stubbs, params.matrixSize, stroutput);
			while(userin != '\n' && userin != 'q') userin = getchar();
			if(userin == 'q') forcequit = TRUE;
		}

		generation_run CUDA_KERNEL_LAUNCH(params.nblocks, params.matrixSize)(d_gpool, d_childpool, d_temppool, d_valpool, d_States);
		CUDA_CHECK_ERROR();
		iterCount += params.iteration_per_run;
		cudaDeviceSynchronize();
		// fetch results from device memory
		cudaMemcpy(h_valpool, d_valpool, valpoolsize, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_gpool, d_gpool, gpoolsize, cudaMemcpyDeviceToHost);

		// check for best solution
		bestVal.fitness = 0;
		bestVal.weight = 0;
		for(i = 0; i < params.nblocks * popCount; i++) {
			if(isBetterSolution(&h_valpool[i], &bestVal)) {
				stubbs.bits = &h_gpool[stubbs.wordCount * i];
				bestVal = h_valpool[i];
			}
		}

		if(params.verbose) {
			// display best solution
			printf("%4d iterations performed. ", iterCount);
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
	if(!params.verbose) {
		printf("%4d iterations performed.\n", iterCount);
		printBitset(&stubbs, stroutput);
	}
	if(bestVal.weight <= inst.capacity && bestVal.fitness > 0) {
		printf("Best found solution: %s\nFitness = %d\nWeight = %d out of %d (%.1f%%)\n",
				stroutput, bestVal.fitness, bestVal.weight, inst.capacity, ((float)(bestVal.weight * 100)) / inst.capacity);
	} else {
		printf("No feasible solution found.\n");
	}

	// cleanup
	cudaFree(d_profit); cudaFree(d_weight);
	cudaFree(d_gpool); cudaFree(d_valpool);
	cudaFree(d_States);
	cudaFree(d_childpool), cudaFree(d_temppool);
	free(h_gpool);
	free(h_valpool);
	destroyInstance(&inst);

	return 0;
}
