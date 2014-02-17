/*
 * bitset.h
 *
 *      Author: Enrico Zamagni
 */

#ifndef BITSET_H_
#define BITSET_H_
#define WORDLEN 32

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef unsigned int word;

typedef struct {
	word *bits;
	unsigned short bitCount;
	unsigned short wordCount;
} Bitset;

int hexprint = 0;

/*
 * Creates a new Bitset with given size.
 * To save performance, the bitset won't be initialized.
 */
Bitset createBitset(int bitnum) {
	Bitset bs;
	bs.wordCount = ceil((float) bitnum / WORDLEN);
	bs.bitCount = bitnum;
	bs.bits = (word*) malloc(bs.wordCount * sizeof(word));

	return bs;
}

/*
 * Frees any data used by the given bitset.
 */
void destroyBitset(Bitset* bs) {
	free(bs->bits);
	free(bs);
}

/*
 * Creates a string representation of the given bitset.
 */
void printBitset(const Bitset* bs, char* strDest) {
	int i, curWord = 0;
	word w, mask = 1u << (WORDLEN - 1);

	if (hexprint) {
		for (curWord = 0; curWord < bs->wordCount; curWord++) {
			strDest += sprintf(strDest, "%08X", bs->bits[curWord]);
		}
	} else {
		mask = 1u << (WORDLEN - 1);
		for (i = 0; i < bs->bitCount; i++) {
			if (i % WORDLEN == 0) {
				w = bs->bits[curWord];
				curWord++;
			}

			*strDest = w & mask ? '1' : '0';
			w <<= 1;
			strDest++;
		}
	}
	*strDest = '\0';
}

/*
 * Sets the specified bit to given value (0 - 1)
 */
void setBit(Bitset* bSet, int pos, int value) {
	int selWord = pos / WORDLEN;
	word mask = 1u << (WORDLEN - pos % WORDLEN - 1);

	if (value) {
		// bit must be set to 1
		bSet->bits[selWord] |= mask;
	} else {
		// bit must be set to 0
		mask = ~mask;
		bSet->bits[selWord] &= mask;
	}
}

/*
 * Retrieves the value of the selected bit
 */
int getBit(Bitset* bSet, int pos) {
	int selWord = pos / WORDLEN;
	word mask = 1u << (WORDLEN - pos % WORDLEN - 1);
	word result = bSet->bits[selWord] & mask;

	return !(result == 0);
}

/*
 * Inverts the value of the specified bit
 */
void flipBit(Bitset* bSet, int pos) {
	int selWord = pos / WORDLEN;
	word mask = 1u << (WORDLEN - pos % WORDLEN - 1);
	bSet->bits[selWord] ^= mask;
}

typedef enum {
	COPY, TOGGLE, SET, RESET
} BitOperation;

/*
 * Sets the value of given bitset by following a format string
 * beginning from the specified bit.
 * N.B. the format string can contain 0s and 1s to force the value
 * of a single bit, '?' to let a bit unchanged and '^' to flip it.
 * Any other character will be interpreted as a '?'.
 * If the input string terminates before the processing of the
 * bitset is finished, all remaining bits will be left untouched.
 */
void setBitString(Bitset* bSet, char* str, int startpos) {
	word tempWord = 0;
	int nbits, nwords, curbit;
	BitOperation bitop;

	for (nwords = 0; nwords < bSet->wordCount; nwords++) {
		if ((nwords + 1) * WORDLEN <= startpos)
			continue;

		for (nbits = 0; nbits < WORDLEN; nbits++) {
			curbit = nwords * WORDLEN + nbits;
			bitop = COPY;
			if (curbit >= startpos && *str != '\0') {
				if (*str == '0')
					bitop = RESET;
				else if (*str == '1')
					bitop = SET;
				else if (*str == '^')
					bitop = TOGGLE;
				str++;
			}

			if ((bitop == COPY && getBit(bSet, curbit) == 1)
					|| (bitop == TOGGLE && getBit(bSet, curbit) == 0))
				bitop = SET;

			if (bitop == SET)
				tempWord += pow(2.0, WORDLEN - nbits - 1);
		}

		bSet->bits[nwords] = tempWord;
		tempWord = 0;
	}
}

#endif
