//ant class

class CAnt
{
public:
    CAnt(void);
    ~CAnt(void);

public:

    int m_nPath[N_CITY_COUNT];
    double m_dbPathLength;
    int m_nAllowedCity[N_CITY_COUNT];
    int m_nCurCityNo;
    int m_nMovedCityCount;

public:

    __device__ void antInit(int antID,float *devData);
    __device__ void antMove(int antID,double *d_Distance,double *d_Trial,float *devData);
    __device__ void antCalPathLength(int antID,double *d_Distance);
    __device__ int antChooseNextCity(int antID,int count,double *d_Distance,double *d_Trial,float *devData);
};

//Constructor
CAnt::CAnt(void)
{
}

//Destructor
CAnt::~CAnt(void)
{
}

__device__
void CAnt:: antInit(int antID,float *devData)
{
    for (int i=0;i<N_CITY_COUNT;i++)
    {
        m_nAllowedCity[i]=1; //设置?¨é.?为没?..è        
        m_nPath[i]=0; //?..走Ã???¨é®¾ç
    }

    //?..走Ã·¯å.å.ç¸º0
    m_dbPathLength=0.0;
    //             //        //?..?..一ä.?..å    
    m_nCurCityNo=(int)N_CITY_COUNT*devData[antID];
    //                         //            //?..?..å?å.è??°ç¸
    m_nPath[0]=m_nCurCityNo;
    //                                     //                    //?..?ºå.?为已ç.è?
    m_nAllowedCity[m_nCurCityNo]=0;
    //                                                 //                            //已Ã»Ã.å.?..ç¸º1
    m_nMovedCityCount=1;
}

__device__
int CAnt::antChooseNextCity(int antID,int count,double *d_Distance,double *d_Trial,float *devData)
{
    int nSelectedCity=-1; //è.ç.ï.?..?..设置ä1

    //==============================================================================
    //    //计ç½..?.??.²¡?»è..å??´ç¿¡æ´.»å
    double dbTotal=0.0,tA,tB;
    double prob[N_CITY_COUNT]; //ä.?.¸ª?.?è.¸­?.??
    //
    for (int i=0;i<N_CITY_COUNT;i++)
    {
        if (m_nAllowedCity[i] == 1) //?.?没å¿
        {    
            //prob[i]=pow(d_Trial[m_nCurCityNo*N_CITY_COUNT+i],ALPHA)*pow(1.0/d_Distance[m_nCurCityNo*N_CITY_COUNT+i],BETA); //该å¸..å.?.??´ç¿¡æ´      
            tA=d_Trial[m_nCurCityNo*N_CITY_COUNT+i];
            tB=1.0/d_Distance[m_nCurCityNo*N_CITY_COUNT+i];
            prob[i]=(tA*tA)*(tB*tB*tB); //ALPHA=2.0,BETA=3.0
            dbTotal=dbTotal+prob[i]; //ç.信æ´.?å.?»å       
        }
        else //å.?.??»èº.??..è.¸­?.??.¼为0
        {
            prob[i]=0.0;
        }
    }

    double dbTemp=0.0;
    if (dbTotal > 0.0)
    {
        dbTemp=dbTotal * devData[m_nMovedCityCount+antID*N_CITY_COUNT-1];

        for (int i=0;i<N_CITY_COUNT;i++)
        {
            if (m_nAllowedCity[i] == 1)
	    {
		dbTemp=dbTemp-prob[i];
		if (dbTemp < 0.0)
		{
		    nSelectedCity=i;
                    break;
                }
            }
        }
    }

    if (nSelectedCity == -1)
    {
        for (int i=0;i<N_CITY_COUNT;i++)
        {
            if (m_nAllowedCity[i] == 1)
	    {
                nSelectedCity=i;
                break;
            }
        }
    }

    return nSelectedCity;
}

__device__
void CAnt::antMove(int antID,double *d_Distance,double *d_Trial,float *devData)
{
    int nCityNo=antChooseNextCity(antID,m_nMovedCityCount,d_Distance,d_Trial,devData);
    m_nPath[m_nMovedCityCount]=nCityNo;
    m_nAllowedCity[nCityNo]=0;
    m_nCurCityNo=nCityNo;
    m_nMovedCityCount++;
}

__device__
void CAnt::antCalPathLength(int antID,double *d_Distance)
{
    m_dbPathLength=0.0;
    int m=0;
    int n=0;

    for (int i=1;i<N_CITY_COUNT;i++)
    {
        m=m_nPath[i];
        n=m_nPath[i-1];
        m_dbPathLength=m_dbPathLength+d_Distance[m*N_CITY_COUNT+n];
    }

    n=m_nPath[0];
    m_dbPathLength=m_dbPathLength+d_Distance[m*N_CITY_COUNT+n];
}
