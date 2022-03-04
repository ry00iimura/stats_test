import numpy as np
import pandas as pd
from scipy.stats import zscore 
from scipy.stats import spearmanr
import scipy.stats
import scipy as sp
import math

def bmTest(x1, x2):
    '''
    Brunner-Munzel
    分布が同じことは仮定せず,二つの確率変数,が同じ分布に従うという帰無仮説を検定する
    '''
    n1, n2 = len(x1), len(x2)
    R = stats.rankdata(list(x1) + list(x2))
    R1, R2 = R[:n1], R[n1:]
    r1_mean, r2_mean = np.mean(R1), np.mean(R2)
    Ri1, Ri2 = stats.rankdata(x1), stats.rankdata(x2)
    var1 = np.var([r - ri for r, ri in zip(R1, Ri1)], ddof=1)
    var2 = np.var([r - ri for r, ri in zip(R2, Ri2)], ddof=1)
    w = ((n1 * n2) * (r2_mean - r1_mean)) / ((n1 + n2) * np.sqrt(n1 * var1 + n2 * var2))
    dof = (n1 * var1 + n2 * var2) ** 2 / ((n1 * var1) ** 2 / (n1 - 1) + (n2 * var2) ** 2 / (n2 - 1))
    c = stats.t.cdf(abs(w), dof) if not np.isinf(w) else 0.0
    p_value = min(c, 1.0 - c) * 2.0
    p = (r2_mean - r1_mean) / (n1 + n2) + 0.5
    return (w, p_value, dof, p)

def unseenDataTest(df,unseenDf):
    '''
    機械学習による予測を行っていると、テストデータやValidデータに対しては精度がでるのに完全な未知データ（unseen）
    に対しては精度がでないケースがある。この場合、そもそも既知のデータと未知のデータが全く同じ母集団から抽出された
    サンプル集団ではない可能性がある。そういったケースで既知のデータと未知のデータが同じ母集団かを検定する。
    実態は「マンホイットニーのU検定」、「Brunner-Munzel検定」および「特徴量の平均値と中央値の比較」
    '''
    from scipy import stats

    MWTest = []
    BMTest = []

    meanFunc = lambda x: x.mean()
    medianFunc = lambda x: x.median()
    varFunc = lambda x: x.var()

    train_means,unseen_means = df.apply(meanFunc),unseenDf.apply(meanFunc)
    train_medians,unseen_medians = df.apply(medianFunc),unseenDf.apply(medianFunc)
    train_vars,unseen_vars = df.apply(varFunc),unseenDf.apply(varFunc)

    for c in df.columns:
        res1 = stats.mannwhitneyu(df[c],unseenDf[c],alternative='two-sided')
        MWTest.append(res1.pvalue)
        res2=bmtest(df[c],unseenDf[c])
        BMTest.append(res2['p_value'])

    return d.DataFrame({
        'U検定のP値':MWTest
        ,'学習データの分散':train_vars
        ,'未知データの分散':unseen_vars
        ,'学習データの平均値':train_means
        ,'未知データの平均値':unseen_means
        ,'学習データの中央値':train_medians
        ,'未知データの中央値':unseen_medians
        ,'Brunner-MunzelのP値':BMTest
        },index=df.columns)    

def chiTest(data):
    'execute chiSquare test'
    x2, p, dof, expected = scipy.stats.chi2_contingency(data)
    print("カイ二乗値は %(x2)s" %locals() )
    print("p値は %(p)s" %locals() )
    print("自由度は %(dof)s" %locals() )
    print( expected )

    if p < 0.05:
        print("有意な差があります")
    else:
        print("有意な差がありません")