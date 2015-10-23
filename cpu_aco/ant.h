#include "tsp.h"

class CAnt
{
public:
    CAnt(void);
    ~CAnt(void);

public:

    int m_nPath[N_CITY_COUNT]; //蚂蚁走的路径
    double m_dbPathLength; //蚂蚁走过的路径长度

    int m_nAllowedCity[N_CITY_COUNT]; //没去过的城市
    int m_nCurCityNo; //当前所在城市编号
    int m_nMovedCityCount; //已经去过的城市数量

public:

    int ChooseNextCity(); //选择下一个城市
    void Init(); //初始化
    void Move(); //蚂蚁在城市间移动
    void Search(); //搜索路径
    void CalPathLength(); //计算蚂蚁走过的路径长度

};
