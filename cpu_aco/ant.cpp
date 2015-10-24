#include <math.h>

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

int CAnt::ChooseNextCity(double g_Trial[N_CITY_COUNT][N_CITY_COUNT], double g_Distance[N_CITY_COUNT][N_CITY_COUNT])
{

    int nSelectedCity = -1; //default -1

    //calculate the total pheromone of current cities and the cities not visited 
    double dbTotal = 0.0;
    double prob[N_CITY_COUNT]; //save the selected probability of each city

    for (int i=0; i<N_CITY_COUNT; i++)
    {
        if (m_nAllowedCity[i] == 1) //the city has not been visited
        {
            //calculate pheromone between this city and current city
            prob[i] = pow(g_Trial[m_nCurCityNo][i], ALPHA) * pow(1.0/g_Distance[m_nCurCityNo][i], BETA);
            dbTotal = dbTotal+prob[i]; //calculate the sum of pheromone
        }
        else //the city has been visited
        {
            prob[i] = 0.0;
        }
    }

    //Roulette wheel selection
    double dbTemp = 0.0;
    if (dbTotal > 0.0) //sum of pheromone > 0
    {
        dbTemp=rnd(0.0, dbTotal); //generate a random between 0 and sum of pheromone

        for (int i=0; i<N_CITY_COUNT; i++)
        {
            if (m_nAllowedCity[i] == 1) //ant has not been to this city
            {
                dbTemp = dbTemp-prob[i]; //Roulette rotating
                if (dbTemp < 0.0) //ant lost in this city
                {
                    nSelectedCity = i;
                    break;
                }
            }
        }
    }

    //if no city is chosen, ant will choose the first city not visited by id order
    if (nSelectedCity == -1)
    {
        for (int i=0; i<N_CITY_COUNT; i++)
        {
            if (m_nAllowedCity[i] == 1) //the city has not been visited
            {
                nSelectedCity=i;
                break;
            }
        }
    }

    //return the id of city which is chosen
    return nSelectedCity;
}

void CAnt::Move(double g_Trial[N_CITY_COUNT][N_CITY_COUNT], double g_Distance[N_CITY_COUNT][N_CITY_COUNT])
{
    int nCityNo = ChooseNextCity(g_Trial, g_Distance); 
    m_nPath[m_nMovedCityCount] = nCityNo; 
    m_nAllowedCity[nCityNo] = 0;
    m_nCurCityNo = nCityNo; 
    m_nMovedCityCount ++; 
}

void CAnt::CalPathLength(double g_Distance[N_CITY_COUNT][N_CITY_COUNT])
{

    m_dbPathLength = 0.0; //default 0
    int m = 0;
    int n = 0;

    for (int i=1; i<N_CITY_COUNT; i++)
    {
        m = m_nPath[i];
        n = m_nPath[i-1];
        m_dbPathLength = m_dbPathLength+g_Distance[m][n];
    }

    //add the distance bewteen first and last city
    n = m_nPath[0];
    m_dbPathLength = m_dbPathLength+g_Distance[m][n];
}

void CAnt::Search(double g_Trial[N_CITY_COUNT][N_CITY_COUNT], double g_Distance[N_CITY_COUNT][N_CITY_COUNT])
{
    Init(); //initialize ants before search

    //ant search until all the cities visited
    while (m_nMovedCityCount < N_CITY_COUNT)
    {
        Move(g_Trial, g_Distance);
    }
    //calculate the tour length
    CalPathLength(g_Distance);
}
