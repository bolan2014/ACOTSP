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

    __device__ void antInit(int antID,float *devRnd);
    __device__ void antMove(int antID,double *d_Distance,double *d_Trial,float *devRnd);
    __device__ void antCalPathLength(int antID,double *d_Distance);
    __device__ int antChooseNextCity(int antID,int count,double *d_Distance,double *d_Trial,float *devRnd);
};

//Constructor
CAnt::CAnt(void)
{
}

//Destructor
CAnt::~CAnt(void)
{
}

//kernel function
__global__
void antSearch_Kernel(CAnt *d_AntAry,double *d_Distance,double *d_Trial,float *devRnd)
{
        int i=threadIdx.x+blockIdx.x*blockDim.x;

        if(i < N_ANT_COUNT)
        {
                d_AntAry[i].antInit(i,devRnd); //initialize data for every ant

                while(d_AntAry[i].m_nMovedCityCount < N_CITY_COUNT)
                {
                        d_AntAry[i].antMove(i,d_Distance,d_Trial,devRnd);
                }
                d_AntAry[i].antCalPathLength(i,d_Distance);
        }
}

__device__
void CAnt:: antInit(int antID,float *devRnd)
{
    for (int i=0;i<N_CITY_COUNT;i++)
    {
        m_nAllowedCity[i]=1; //设置?¨é.?为没?..è        
        m_nPath[i]=0; //?..走Ã???¨é®¾ç
    }

    m_dbPathLength=0.0;
    //
    m_nCurCityNo=(int)N_CITY_COUNT*devRnd[antID];

    m_nPath[0]=m_nCurCityNo;

    m_nAllowedCity[m_nCurCityNo]=0;

    m_nMovedCityCount=1;
}

__device__
int CAnt::antChooseNextCity(int antID,int count,double *d_Distance,double *d_Trial,float *devRnd)
{
    int nSelectedCity=-1; 

    //==============================================================================
    double dbTotal=0.0, tA, tB;
    double prob[N_CITY_COUNT];
    //
    for (int i=0;i<N_CITY_COUNT;i++)
    {
        if (m_nAllowedCity[i] == 1) 
        {    
            //prob[i]=pow(d_Trial[m_nCurCityNo*N_CITY_COUNT+i],ALPHA)*pow(1.0/d_Distance[m_nCurCityNo*N_CITY_COUNT+i],BETA); //该å¸..å.?.??´ç¿¡æ´      
            tA=d_Trial[m_nCurCityNo*N_CITY_COUNT+i];
            tB=1.0/d_Distance[m_nCurCityNo*N_CITY_COUNT+i];
            prob[i]=(tA*tA)*(tB*tB*tB); //ALPHA=2.0,BETA=3.0
            dbTotal=dbTotal+prob[i]; 
        }
        else 
        {
            prob[i]=0.0;
        }
    }

    double dbTemp=0.0;
    if (dbTotal > 0.0)
    {
        dbTemp=dbTotal * devRnd[m_nMovedCityCount+antID*N_CITY_COUNT-1];

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
void CAnt::antMove(int antID,double *d_Distance,double *d_Trial,float *devRnd)
{
    int nCityNo=antChooseNextCity(antID,m_nMovedCityCount,d_Distance,d_Trial,devRnd);
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
