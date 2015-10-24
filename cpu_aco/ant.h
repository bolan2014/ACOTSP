const double ALPHA = 2.0;
const double BETA = 3.0;

const int N_CITY_COUNT = 51;

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

    int ChooseNextCity(double g_Trial[N_CITY_COUNT][N_CITY_COUNT], double g_Distance[N_CITY_COUNT][N_CITY_COUNT]); //ants choose next city
    void Init(); //initial ants'info
    void Move(double g_Trial[N_CITY_COUNT][N_CITY_COUNT], double g_Distance[N_CITY_COUNT][N_CITY_COUNT]); //ants'move
    void Search(); //ants search path
    void CalPathLength(double g_Distance[N_CITY_COUNT][N_CITY_COUNT]); //calculate length of path

};
