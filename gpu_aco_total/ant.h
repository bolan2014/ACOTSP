//ant class

const double DBQ = 100.0; //total pheromone

const double ROU = 0.9; //param for evaporation

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
    double deposit;

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

//ant search kernel
__global__
void antSearch_Kernel(CAnt *d_AntAry,double *d_Distance,double *d_Trial,float *devData)
{
    int i=threadIdx.x+blockIdx.x*blockDim.x;

    if(i < N_ANT_COUNT)
    {
        d_AntAry[i].antInit(i,devData); //initialize data for every ant

        while(d_AntAry[i].m_nMovedCityCount < N_CITY_COUNT)
        {
            d_AntAry[i].antMove(i,d_Distance,d_Trial,devData);
        }

        d_AntAry[i].antCalPathLength(i,d_Distance);
        d_AntAry[i].deposit = DBQ / d_AntAry[i].m_dbPathLength;
    }
}

//kernel for evaporation
__global__
void evaporateTrial_Kernel(double *d_Trial)
{
    int i = threadIdx.x+blockIdx.x*blockDim.x;

    if(i < N_CITY_COUNT * N_CITY_COUNT)
    {
        d_Trial[i] = d_Trial[i] * ROU;
    }
}

//kernel for strengthen
__global__
void enhanceTrial_Kernel(CAnt *d_AntAry,double *d_Trial)
{
	int i = threadIdx.x+blockIdx.x*blockDim.x;

	int m, n;
	if(i < 1)
	{
        for(int k=1; k<N_ANT_COUNT; k++)
        {
            if(d_AntAry[k].m_dbPathLength < d_AntAry[0].m_dbPathLength)
                d_AntAry[0] = d_AntAry[k];
        }
    
        for(int j = 1; j < N_CITY_COUNT; ++j)
        {
            m = d_AntAry[0].m_nPath[j];
            n = d_AntAry[0].m_nPath[j-1];
 
            d_Trial[n*N_CITY_COUNT+m] += d_AntAry[0].deposit;
            d_Trial[m*N_CITY_COUNT+n] = d_Trial[n*N_CITY_COUNT+m];
        }
 
        n = d_AntAry[i].m_nPath[0];
        d_Trial[n*N_CITY_COUNT+m] += d_AntAry[0].deposit;
        d_Trial[m*N_CITY_COUNT+n] = d_Trial[n*N_CITY_COUNT+m];
    }
}

__device__
void CAnt:: antInit(int antID,float *devData)
{
    for (int i=0;i<N_CITY_COUNT;i++)
    {
        m_nAllowedCity[i]=1; //set all cities as not have been to
        m_nPath[i]=0; //initial path as 0
    }

    m_dbPathLength=0.0;
    //
    m_nCurCityNo=(int)N_CITY_COUNT*devData[antID];

    m_nPath[0]=m_nCurCityNo;

    m_nAllowedCity[m_nCurCityNo]=0;

    m_nMovedCityCount=1;
}

__device__
int CAnt::antChooseNextCity(int antID,int count,double *d_Distance,double *d_Trial,float *devData)
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
            //prob[i]=pow(d_Trial[m_nCurCityNo*N_CITY_COUNT+i],ALPHA)*pow(1.0/d_Distance[m_nCurCityNo*N_CITY_COUNT+i],BETA); 
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
