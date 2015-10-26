#include <curand.h>

//for parallel cuRand
float *devData;

const int N_ANT_COUNT = 256;
const int N_CITY_COUNT = 51; 
const int N_IT_COUNT=15;

//tsp file name
const char *tspFile = "tsp/eil51.tsp";

void dev_rnd(unsigned int nSeed)
{
        curandGenerator_t gen;

        cudaMalloc((void **)&devData, N_ANT_COUNT*N_CITY_COUNT*sizeof(float));

        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, nSeed);
        curandGenerateUniform(gen, devData, N_ANT_COUNT*N_CITY_COUNT);

        curandDestroyGenerator(gen);
}
