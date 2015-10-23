#include "ant.h"
#include "rand.h"

//constructor
CAnt::CAnt(void)
{
}

//destructor
CAnt::~CAnt(void)
{
}

void CAnt::Init()
{

    for (int i=0; i<N_CITY_COUNT; i++)
    {
        m_nAllowedCity[i] = 1; //Set all the cities as not have been to
        m_nPath[i] = 0; //initial path as 0
    }

    //initial length of path as 0
    m_dbPathLength = 0.0;

    //select a start city with random
    m_nCurCityNo=rnd(0, N_CITY_COUNT);
    
    //save the start city into the path array
    m_nPath[0] = m_nCurCityNo;
    
    //identify the start city as already been to
    m_nAllowedCity[m_nCurCityNo] = 0;
    
    //set the number of city visited as 1
    m_nMovedCityCount = 1;
}
