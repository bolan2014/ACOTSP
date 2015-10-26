#include <iostream>
#include <stdio.h>
#include <math.h>
#include <time.h>
//#include <curand.h>

#include "rand.h"
#include "ant.h"

//const double ALPHA=2.0; 
//const double BETA=3.0;
const double ROU=0.5; //信息素残留参数

const double DBQ=100.0; //总的信息素
const double DB_MAX=10e9; //一个标志数，10的9次方

double g_Trial[N_CITY_COUNT][N_CITY_COUNT]; //两两城市间信息素，就是环境信息素
double g_Distance[N_CITY_COUNT][N_CITY_COUNT]; //两两城市间距离

//data on device
double *d_Distance,*d_Trial;

//tsp城市坐标数据
double x_Ary[N_CITY_COUNT],y_Ary[N_CITY_COUNT];

//返回浮点数四舍五入取整后的浮点数
double ROUND(double dbA)
{
    return (double)((int)(dbA+0.5));
}


//tsp class
class CTsp
{
public:
    CTsp(void);
    ~CTsp(void);

public:
    CAnt m_cAntAry[N_ANT_COUNT]; //蚂蚁数组(host)
    CAnt m_cBestAnt; //定义一个蚂蚁变量，用来保存搜索过程中的最优结果
                                        //该蚂蚁不参与搜索，只是用来保存最优结果

public:

    //get city position
    void readTsp();

    //初始化数据
    void InitData();

    //开始搜索
    void Search();

    //更新环境信息素
    void UpdateTrial();

    //搜索路径
    void antSearch(); //kernel inside

};

//构造函数
CTsp::CTsp(void)
{
}

CTsp::~CTsp(void)
{
}

//read tsp file
void CTsp::readTsp()
{
	int i,j;

	FILE *fp=fopen(tspFile, "r") ;
	if(fp == NULL)
	{
		printf("sorry,file not found!\n");
		exit(0);
	}

	for(i=0;i<N_CITY_COUNT;i++)
	{
		fscanf(fp,"%d%lf%lf",&j,&x_Ary[i],&y_Ary[i]);
	}

	fclose(fp);
}

CAnt *d_AntAry; //ants on GPU

//初始化数据
void CTsp::InitData()
{
    //read tsp file
    readTsp();

    //先把最优蚂蚁的路径长度设置成一个很大的值
    m_cBestAnt.m_dbPathLength=DB_MAX;

    //计算两两城市间距离
    double dbTemp=0.0;
    for (int i=0;i<N_CITY_COUNT;i++)
    {
        for (int j=0;j<N_CITY_COUNT;j++)
        {
        	//compute distance between cities
            dbTemp=(x_Ary[i]-x_Ary[j])*(x_Ary[i]-x_Ary[j])+(y_Ary[i]-y_Ary[j])*(y_Ary[i]-y_Ary[j]);
            dbTemp=sqrt(dbTemp);
            g_Distance[i][j]=ROUND(dbTemp);
        }
    }

    //初始化环境信息素，先把城市间的信息素设置成一样
    //这里设置成1.0，设置成多少对结果影响不是太大，对算法收敛速度有些影响
    for (int i=0;i<N_CITY_COUNT;i++)
    {
        for (int j=0;j<N_CITY_COUNT;j++)
        {
            g_Trial[i][j]=1.0;
        }
    }
}

// every ant search as a thread
void CTsp::antSearch()
{
    //initialize data on GPU
    size_t size=sizeof(CAnt)*N_ANT_COUNT;
    cudaMalloc(&d_AntAry,size);

    size=sizeof(double)*N_CITY_COUNT*N_CITY_COUNT;
    cudaMalloc(&d_Distance,size);
    cudaMalloc(&d_Trial,size);

    //data copy
    cudaMemcpy(d_Distance,&g_Distance[0][0],size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_Trial,&g_Trial[0][0],size,cudaMemcpyHostToDevice);

    //kernel use
    antSearch_Kernel<<<ceil(N_ANT_COUNT/256.0), 256.0>>>(d_AntAry,d_Distance,d_Trial,devData);

    size=sizeof(CAnt)*N_ANT_COUNT;
    cudaMemcpy(m_cAntAry, &d_AntAry[0],size,cudaMemcpyDeviceToHost);
}

//更新环境信息素
void CTsp::UpdateTrial()
{
    //临时数组，保存各只蚂蚁在两两城市间新留下的信息素
    double dbTempAry[N_CITY_COUNT][N_CITY_COUNT];
    memset(dbTempAry,0,sizeof(dbTempAry)); //先全部设置为0

    //计算新增加的信息素,保存到临时数组里
    int m=0;
    int n=0;
    for (int i=0;i<N_ANT_COUNT;i++) //计算每只蚂蚁留下的信息素
    {
            for (int j=1;j<N_CITY_COUNT;j++)
            {
                m=m_cAntAry[i].m_nPath[j];
                n=m_cAntAry[i].m_nPath[j-1];
                dbTempAry[n][m]=dbTempAry[n][m]+DBQ/m_cAntAry[i].m_dbPathLength;
                dbTempAry[m][n]=dbTempAry[n][m];
            }

            //最后城市和开始城市之间的信息素
            n=m_cAntAry[i].m_nPath[0];
            dbTempAry[n][m]=dbTempAry[n][m]+DBQ/m_cAntAry[i].m_dbPathLength;
            dbTempAry[m][n]=dbTempAry[n][m];

    }

    //==================================================================
    //更新环境信息素
    for (int i=0;i<N_CITY_COUNT;i++)
    {
        for (int j=0;j<N_CITY_COUNT;j++)
        {
            g_Trial[i][j]=g_Trial[i][j]*ROU+dbTempAry[i][j]; //最新的环境信息素 = 留存的信息素 + 新留下的信息素
        }
    }
}

void CTsp::Search()
{

    char cBuf[256]; //打印信息用

    {
        //每只蚂蚁搜索一遍
        antSearch();

        //保存最佳结果
        for (int j=0;j<N_ANT_COUNT;j++)
        {
            if (m_cAntAry[j].m_dbPathLength < m_cBestAnt.m_dbPathLength)
            {
                m_cBestAnt=m_cAntAry[j];
            }
        }

        //更新环境信息素
        UpdateTrial();

        //输出目前为止找到的最优路径的长度
        sprintf(cBuf,"\n[%d] %.0f",i+1,m_cBestAnt.m_dbPathLength);
        printf(cBuf);
    }
}

int main()
{
    printf("Ants start searching tours . . .");

    //count running time
    clock_t MyTime, UTime;
    double duration = 0.0;
    MyTime = clock();

    //用当前时间点初始化随机种子，防止每次运行的结果都相同
    time_t tm;
    time(&tm);
    unsigned int nSeed=(unsigned int)tm;

    //generate random number for next city
    dev_rnd(nSeed);

    //开始搜索
    CTsp tsp;

    tsp.InitData(); //初始化
    tsp.Search(); //开始搜索

    //输出结果
    /*printf("\nThe best tour is :\n");

    char cBuf[128];
    for (int i=0;i<N_CITY_COUNT;i++)
    {
        sprintf(cBuf,"d ",tsp.m_cBestAnt.m_nPath[i]+1);
        if (i % 20 == 0)
        {
            printf("\n");
        }
        printf(cBuf);
    }*/

    UTime = clock();
    duration = (double)(UTime - MyTime) / CLOCKS_PER_SEC;
    printf("\nTotal time is %0.3f seconds\n", duration);
    printf("\nAnts' searching is done!\n");

    //release memory on device
    cudaFree(devData);
    cudaFree(d_Distance);
    cudaFree(d_Trial);
    cudaFree(d_AntAry);

    return 0;
}
