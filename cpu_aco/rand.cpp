#include <iostream>

#include "rand.h"

int rnd(int nLow,int nUpper)
{
    return (int)(nLow+(nUpper-nLow)*rand()/((double)RAND_MAX+1.0));
}
