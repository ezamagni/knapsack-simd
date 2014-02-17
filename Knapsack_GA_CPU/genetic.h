/*
 * genetic.h
 *
 *      Author: Enrico Zamagni
 */

#ifndef GENETIC_H_
#define GENETIC_H_

#include "utils.h"

typedef struct {
	Bitset bitset;
	SolutionVal value;
} Individual;

/*
 * Instantiates a new BitSet representing an individual
 * with given chromosome length. The chromosome will be
 * randomly created.
 * N.B.	The individual will NOT be evaluated and no test
 *  will be performed to check if the returned
 *  individual represent a feasible solution.
 */
void createIndividual(Individual *newind, int clen) {
	newind->bitset = createBitset(clen);

	// populate bitset with random bytes
	int i;
	for (i = 0; i < newind->bitset.wordCount; i++)
		newind->bitset.bits[i] = random() & 0xffffffff;
}

/*
 * Instantiates a new BitSet representing an individual
 * with given chromosome length. The chromosome will be
 * set to zero.
 * N.B.	The individual will NOT be evaluated and no test
 *  will be performed to check if the returned
 *  individual represent a feasible solution.
 */
void createEmptyIndividual(Individual *newind, int clen) {
	newind->bitset = createBitset(clen);

	// populate bitset with random bytes
	int i;
	for (i = 0; i < newind->bitset.wordCount; i++)
		newind->bitset.bits[i] = 0;
}

/*
 * Copies the cromosome information of an individual to another.
 */
void copyIndividual(const Individual *source, Individual *destination) {
	int w;
	for(w = 0; w < source->bitset.wordCount; w++)
		destination->bitset.bits[w] = source->bitset.bits[w];
}

/*
 * Performs a simple one-point crossover between two individuals
 * at the given pointcut. The resulting child will be returned into
 * the provided individual stub.
 * N.B. The resuting individual will NOT be evaluated.
 */
void performCrossoverOnePoint(Individual *parent, Individual *partner,
		Individual *child, int pcut, int direction) {
	int bitCopied = 0;

	// create mask
	word mask = 0xffffffff << (WORDLEN - (pcut % WORDLEN));

	word *pw1 = parent->bitset.bits, *pw2 = partner->bitset.bits;
	word *cw = child->bitset.bits;
	// transfer genes to offspring..
	while (bitCopied < parent->bitset.bitCount) {
		if (bitCopied + WORDLEN < pcut) {
			// ..before the cut
			if(direction) *cw = *pw1; else *cw = *pw2;
		} else if (bitCopied > pcut) {
			//.. after the cut
			if(direction) *cw = *pw2; else *cw = *pw1;
		} else {
			// ..into the cut
			if(direction) *cw = (*pw1 & mask) | (*pw2 & ~mask);
			else *cw = (*pw2 & mask) | (*pw1 & ~mask);
		}
		pw1++; pw2++;
		cw++;
		bitCopied += WORDLEN;
	}
}

void performMutation(Individual *subject, int pos) {
	int selWord = pos / WORDLEN;
	word mask = 1u << (WORDLEN - pos % WORDLEN - 1);
	subject->bitset.bits[selWord] ^= mask;
}

/*
 * Evaluates both individual's fitness and weight using two
 * given arrays containing profit and weight values
 */
void evaluateIndividual(Individual *subject, const KPInstance pinst, const float pratio) {
	int b, curWord = 0;
	word w, mask = 1u << (WORDLEN - 1);

	subject->value.fitness = 0;
	subject->value.weight = 0;

	// process every bit
	for (b = 0; b < subject->bitset.bitCount; b++) {
		if (b % WORDLEN == 0) {
			w = subject->bitset.bits[curWord];
			curWord++;
		}
		if (w & mask) {
			subject->value.fitness += pinst.profit[b];
			subject->value.weight += pinst.weight[b];
		}
		w <<= 1;
	}

	// penalize fitness if unfeasible
	if(subject->value.weight > pinst.capacity) {
		int overweight = subject->value.weight - pinst.capacity;
		subject->value.fitness -= overweight * pratio;
	}
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

void popdump(Individual *popaddr, dim3 popsize, char *stroutput) {
	int x, y;
	for(y = 0; y < popsize.y; y++) {
		for(x = 0; x < popsize.x; x++) {
			printBitset(&popaddr->bitset, stroutput);
			printf("%s ", stroutput);
			popaddr++;
		}
		printf("\n");
	}
	printf("\n");
}

#endif /* GENETIC_H_ */
