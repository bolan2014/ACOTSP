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
