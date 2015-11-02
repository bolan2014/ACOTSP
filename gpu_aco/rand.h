#include <curand.h>

//for parallel cuRand
float *devRnd;

const int N_ANT_COUNT = 1024;
const int N_CITY_COUNT = 493; 
const int N_IT_COUNT = 15;

//tsp file name
const char *tspFile = "tsp/d493.tsp";

void dev_rnd(unsigned int nSeed)
{
        curandGenerator_t gen;

        cudaMalloc((void **)&devRnd, N_ANT_COUNT*N_CITY_COUNT*sizeof(float));

        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, nSeed);
        curandGenerateUniform(gen, devRnd, N_ANT_COUNT*N_CITY_COUNT);

        curandDestroyGenerator(gen);
}
