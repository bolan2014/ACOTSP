#include <iostream>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "rand.h"
#include "ant.h"

const double ALPHA=2.0; //启发因子，信息素的重要程度
const double BETA=3.0;   //期望因子，城市间距离的重要程度
const double ROU=0.5; //信息素残留参数

const int N_ANT_COUNT=256; //蚂蚁数量
const int N_IT_COUNT=15; //迭代次数

const double DBQ=100.0; //总的信息素
const double DB_MAX=10e9; //一个标志数，10的9次方

double g_Trial[N_CITY_COUNT][N_CITY_COUNT]; //两两城市间信息素，就是环境信息素
double g_Distance[N_CITY_COUNT][N_CITY_COUNT]; //两两城市间距离

//tsp城市坐标数据
double x_Ary[N_CITY_COUNT],y_Ary[N_CITY_COUNT];

//返回浮点数四舍五入取整后的浮点数
double ROUND(double dbA)
{
    return (double)((int)(dbA+0.5));
}

//选择下一个城市
//返回值 为城市编号
int CAnt::ChooseNextCity()
{

    int nSelectedCity=-1; //返回结果，先暂时把其设置为-1

    //==============================================================================
    //计算当前城市和没去过的城市之间的信息素总和

    double dbTotal=0.0;
    double prob[N_CITY_COUNT]; //保存各个城市被选中的概率

    for (int i=0;i<N_CITY_COUNT;i++)
    {
        if (m_nAllowedCity[i] == 1) //城市没去过
        {
            prob[i]=pow(g_Trial[m_nCurCityNo][i],ALPHA)*pow(1.0/g_Distance[m_nCurCityNo][i],BETA); //该城市和当前城市间的信息素
            dbTotal=dbTotal+prob[i]; //累加信息素，得到总和
        }
        else //如果城市去过了，则其被选中的概率值为0
        {
            prob[i]=0.0;
        }
    }

    //==============================================================================
    //进行轮盘选择
    double dbTemp=0.0;
    if (dbTotal > 0.0) //总的信息素值大于0
    {
        dbTemp=rnd(0.0,dbTotal); //取一个随机数

        for (int i=0;i<N_CITY_COUNT;i++)
        {
            if (m_nAllowedCity[i] == 1) //城市没去过
            {
                dbTemp=dbTemp-prob[i]; //这个操作相当于转动轮盘，如果对轮盘选择不熟悉，仔细考虑一下
                if (dbTemp < 0.0) //轮盘停止转动，记下城市编号，直接跳出循环
                {
                    nSelectedCity=i;
                    break;
                }
            }
        }
    }

    //==============================================================================
    //如果城市间的信息素非常小 ( 小到比double能够表示的最小的数字还要小 )
    //那么由于浮点运算的误差原因，上面计算的概率总和可能为0
    //会出现经过上述操作，没有城市被选择出来
    //出现这种情况，就把第一个没去过的城市作为返回结果

    //题外话：刚开始看的时候，下面这段代码困惑了我很长时间，想不通为何要有这段代码，后来才搞清楚。
    if (nSelectedCity == -1)
    {
        for (int i=0;i<N_CITY_COUNT;i++)
        {
            if (m_nAllowedCity[i] == 1) //城市没去过
            {
                nSelectedCity=i;
                break;
            }
        }
    }

    //==============================================================================
    //返回结果，就是城市的编号
    return nSelectedCity;
}


//蚂蚁在城市间移动
void CAnt::Move()
{
    int nCityNo=ChooseNextCity(); //选择下一个城市

    m_nPath[m_nMovedCityCount]=nCityNo; //保存蚂蚁走的路径
    m_nAllowedCity[nCityNo]=0;//把这个城市设置成已经去过了
    m_nCurCityNo=nCityNo; //改变当前所在城市为选择的城市
    m_nMovedCityCount++; //已经去过的城市数量加1
}

//蚂蚁进行搜索一次
void CAnt::Search()
{
    Init(); //蚂蚁搜索前，先初始化

    //如果蚂蚁去过的城市数量小于城市数量，就继续移动
    while (m_nMovedCityCount < N_CITY_COUNT)
    {
        Move();
    }

    //完成搜索后计算走过的路径长度
    CalPathLength();
}


//计算蚂蚁走过的路径长度
void CAnt::CalPathLength()
{

    m_dbPathLength=0.0; //先把路径长度置0
    int m=0;
    int n=0;

    for (int i=1;i<N_CITY_COUNT;i++)
    {
        m=m_nPath[i];
        n=m_nPath[i-1];
        m_dbPathLength=m_dbPathLength+g_Distance[m][n];
    }

    //加上从最后城市返回出发城市的距离
    n=m_nPath[0];
    m_dbPathLength=m_dbPathLength+g_Distance[m][n];

}


//tsp类
class CTsp
{
public:
    CTsp(void);
    ~CTsp(void);

public:
    CAnt m_cAntAry[N_ANT_COUNT]; //蚂蚁数组
    CAnt m_cBestAnt; //定义一个蚂蚁变量，用来保存搜索过程中的最优结果
                                        //该蚂蚁不参与搜索，只是用来保存最优结果

public:

    //city position
    void readTsp();

    //初始化数据
    void InitData();

    //开始搜索
    void Search();

    //更新环境信息素
    void UpdateTrial();

};


//构造函数
CTsp::CTsp(void)
{
}

CTsp::~CTsp(void)
{
}

void CTsp::readTsp()
{
	int i,j;

	FILE *fp=fopen("tsp/eil51.tsp","r") ;
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
            dbTemp=(x_Ary[i]-x_Ary[j])*(x_Ary[i]-x_Ary[j])+(y_Ary[i]-y_Ary[j])*(y_Ary[i]-y_Ary[j]);
            dbTemp=pow(dbTemp,0.5);
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

    //在迭代次数内进行循环
    for (int i=0;i<N_IT_COUNT;i++)
    {
        //每只蚂蚁搜索一遍
        for (int j=0;j<N_ANT_COUNT;j++)
        {
            m_cAntAry[j].Search();
        }

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
    srand(nSeed);

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

    return 0;
}
