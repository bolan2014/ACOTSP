#include "tsp.h"

class CAnt
{
  public:
    CAnt(void);
    ~CAnt(void);

  public:

    int m_nPath[N_CITY_COUNT]; //tour of an ant
    double m_dbPathLength; //tour length

    int m_nAllowedCity[N_CITY_COUNT]; //cities have not been visited
    int m_nCurCityNo; //id of the current city
    int m_nMovedCityCount; //number if cities already visited

  public:

    int ChooseNextCity(); //ants choose next city
    void Init(); //initial ants'info
    void Move(); //ants'move
    void Search(); //ants search path
    void CalPathLength(); //calculate length of path

};
