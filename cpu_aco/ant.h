const int N_CITY_COUNT = 318; //number of cities
const int N_ANT_COUNT = 1024; //number of ants
const int N_IT_COUNT = 15; //number of iterations

const double ALPHA = 2.0;
const double BETA = 3.0;
const double ROU = 0.5;

const double DBQ=100.0;
const double DB_MAX=10e9;

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
    
    //ants choose next city
    int ChooseNextCity(double g_Trial[N_CITY_COUNT][N_CITY_COUNT], double g_Distance[N_CITY_COUNT][N_CITY_COUNT]);
    
    //initial ants'info
    void Init();
    
    //ants'move
    void Move(double g_Trial[N_CITY_COUNT][N_CITY_COUNT], double g_Distance[N_CITY_COUNT][N_CITY_COUNT]);
    
    //ants search path
    void Search(double g_Trial[N_CITY_COUNT][N_CITY_COUNT], double g_Distance[N_CITY_COUNT][N_CITY_COUNT]);

    //calculate length of path
    void CalPathLength(double g_Distance[N_CITY_COUNT][N_CITY_COUNT]);
};
