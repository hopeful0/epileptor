# 排序评价标准

# 参考:
# https://www.cnblogs.com/startover/p/3141616.html
# https://blog.csdn.net/weixin_38405636/article/details/80675312
# https://blog.csdn.net/hero00e/article/details/87909790

import numpy as np

# Mean reciprocal rank(MRR) 平均倒排序值

# Mean average precision(MAP) 平均准确率

def mean_average_precision(standard, subject):
    if not isinstance(standard, np.ndarray):
        standard = np.array(standard)
    j = 1
    s = 0
    for i in range(len(subject)):
        y = np.where(standard == subject[i])[0]
        if len(y) > 0:
            s += j / (i + 1)
            j += 1
    return s / len(standard)

# Discounted cumulative gain(DCG)
def DCG(standard, subject, log_gain=True):
    l = len(standard)
    gains = [2**i-1 for i in range(l, -1, -1)] if log_gain else list(range(l, -1, -1))

    vdcg = 0
    if not isinstance(subject, np.ndarray):
        subject = np.array(subject)
    for i in range(len(subject)):
        x = subject[i]
        y = np.where(standard == x)[0]
        y = l if len(y) == 0 else y[0]
        vdcg += gains[y] / np.log2(i + 2)
    return vdcg

# Normalized discounted cumulative gain(NDCG)

# use `standard` as iNCG
def NDCG_V2(standard, subject, kw={}):
     return DCG(standard, subject, **kw) / DCG(standard, standard, **kw)

# standard, rearrangement subject as iNCG
def NDCG(standard, subject, kw={}):
    isubject = np.array(standard)[np.isin(standard, subject)]
    if (len(isubject)<=0):
        return 0
    return DCG(standard, subject, **kw) / DCG(standard, isubject, **kw)

# Rank correlation(RC)
# https://en.wikipedia.org/wiki/Rank_correlation

# convert vector to rank, value 0 as a single rank
# order: small to big
def vec2rank(vec):
    if not vec is np.ndarray:
        vec = np.array(vec)
    rank = np.zeros_like(vec)
    rank[vec>0] = np.argsort(vec[vec>0]) + 1
    rank = np.max(rank) - rank
    return rank

# Spearman's ρ
# https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
def Spearman(v1, v2):
    r1 = vec2rank(v1)
    r2 = vec2rank(v2)
    cov = np.cov([r1, r2])
    cov = cov[0, 1]
    return cov / (np.std(r1, ddof=1) * np.std(r2, ddof=1))

# Kendall's τ
# https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
def Kendall(v1, v2):
    r1 = vec2rank(v1)
    r2 = vec2rank(v2)
    n = len(r1)
    tau = 0
    for i in range(n):
        for j in range(i + 1, n):
            tau += np.sign(r1[i] - r1[j]) * np.sign(r2[i] - r2[j])
    return tau * 2 / (n * (n - 1))


# dynamic time warping(DTW)
def distance(w1,w2):
    d = abs(w2 - w1)
    return d

def DTW(s1,s2):
    m = len(s1)
    n = len(s2)

    # 构建二位dp矩阵,存储对应每个子问题的最小距离
    dp = [[0]*n for _ in range(m)] 

    # 起始条件,计算单个字符与一个序列的距离
    for i in range(m):
        dp[i][0] = distance(s1[i],s2[0])
    for j in range(n):
        dp[0][j] = distance(s1[0],s2[j])
    
    # 利用递推公式,计算每个子问题的最小距离,矩阵最右下角的元素即位最终两个序列的最小值
    for i in range(1,m):
        for j in range(1,n):
            dp[i][j] = min(dp[i-1][j-1],dp[i-1][j],dp[i][j-1]) + distance(s1[i],s2[j])
    
    return dp[-1][-1]