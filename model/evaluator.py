import numpy as np
import pandas as pd
from IPython.core.debugger import set_trace
from sklearn import linear_model
from sklearn.metrics import r2_score
from numba import jit, float64


# 종목별 CAGR
@jit(float64(float64[:]), nopython=True)#, fastmath=True)
def _cagr(p):
    return ((p[-1]/p[0])**(250/(len(p)-1)) - 1) * 100


@jit(float64(float64[:]), nopython=True)#, fastmath=True)
def _std(p):
    rt = p[1:]/p[:-1] - 1.0
    n = len(rt)
    rt_mean = rt.sum() / n
    out = (((rt-rt_mean)**2).sum() / (n-1))**0.5
    return out * (250**0.5) * 100


# 종목별 Sharpe
@jit(float64(float64[:]), nopython=True)#, fastmath=True)
def _sharpe(p): 
    std = _std(p)
    if std==0:
        return np.nan
    else:
        return _cagr(p)/std
    

# 종목별 변동성
#def _std2(p): 
#    return np.nanstd(p[1:]/p[:-1]-1) * (250**0.5) * 100
    
    
# 종목별 하방변동성
def _std_down(p):
    r = p[1:]/p[:-1]-1
    r = r[~np.isnan(r)]
    r_neg = r[r<0]
    return ((((r_neg**2).sum() / (len(r)-1))**0.5) * (250**0.5)) * 100


def _std_dir_by_r(r, dir_): 
    if dir_==1:
        r_dir = r[r>0]
    elif dir_==-1:    
        r_dir = r[r<0]
    return ((((r_dir**2).sum() / (len(r)-1))**0.5) * (250**0.5)) * 100


def _std_dir(p, dir_): 
    r = p[1:]/p[:-1]-1
    r = r[~np.isnan(r)]
    if dir_==1:
        r_dir = r[r>0]
    elif dir_==-1:    
        r_dir = r[r<0]
    return ((((r_dir**2).sum() / (len(r)-1))**0.5) * (250**0.5)) * 100      


# 종목별 MDD
def _mdd(p):
    return (p/np.maximum.accumulate(p)-1).min() * 100


# 종목별 수익안정성
def _consistency(p):
    if len(p)==0:
        return np.nan

    y = np.log(p)
    X = np.arange(len(p)).reshape(-1,1)
    model = linear_model.LinearRegression(fit_intercept=False).fit(X, y)
    return r2_score(y, model.predict(X))


# 종목별 베타
def _beta(rtns):
    if len(rtns)==0:
        return np.nan

    y = rtns[:,0].reshape(-1,1)
    X = rtns[:,1].reshape(-1,1)
    model = linear_model.LinearRegression().fit(X, y)
    return model.coef_[0,0]


def _turnover(weights):
    turnover = pd.Series()

    for idt, dt in list(enumerate(weights.index))[1:]:
        turnover[dt] = weights.iloc[idt].sub(weights.iloc[idt-1], fill_value=0).abs().sum()/2  

    return turnover.rolling(12).sum().dropna()


def _stats(cum, beta_to, n_roll_stats, start=None, end=None):
    cum_ = cum.copy()
    
    if start is not None:
        cum_ = cum_.loc[start:]
        cum_ /= cum_.iloc[0]
        
    if end is not None:
        cum_ = cum_.loc[:end]
    
    cum_last = cum_.iloc[-1]
    n_samples = cum_.count()

    # Base stats: 전체구간
    cagr = (cum_last**(250/(n_samples-1)) - 1) * 100
    std = cum_.pct_change().std() * (250**0.5) * 100
    sharpe = cagr / std
    #mdd = (cum.expanding().apply(lambda x: x[-1]/np.nanmax(x)-1).min()) * 100
    mdd = (cum_.div(cum_.cummax()) - 1).min() * 100
    #set_trace()
    consistency = (pd.Series([_consistency(cum_[col].dropna()) for col in cum_], index=cum_.columns)) * 100
    beta = pd.Series([_beta(cum_[[col, beta_to]].pct_change().dropna(how='any').values) for col in cum_], index=cum_.columns)

    # Rolling stats
    #set_trace()
    cum_roll = cum_.rolling(n_roll_stats)
    cagr_roll = cum_roll.apply(_cagr, raw=True)
    std_roll = cum_roll.apply(_std, raw=True)
    sharpe_roll = cagr_roll/std_roll
    
    cagr_roll_med = cagr_roll.median()
    std_roll_med = std_roll.median()
    sharpe_roll_med = sharpe_roll.median()    
    loss_proba = (cagr_roll<0).sum()/cagr_roll.count() * 100

    # With 1M returns
    r_month = cum_.resample('M').ffill().pct_change()
    hit = ((r_month>0).sum() / (r_month.count()-1)) * 100
    profit_to_loss = - r_month[r_month>0].mean() / r_month[r_month<0].mean()

    return pd.DataFrame({
        'cum_last': cum_last,
        'n_samples': n_samples, 
        'cagr': cagr, 
        'std': std,
        'sharpe': sharpe,
        'mdd': mdd,
        'cagr_roll_med': cagr_roll_med, 
        'std_roll_med': std_roll_med, 
        'sharpe_roll_med': sharpe_roll_med, 
        'beta': beta, 
        'loss_proba': loss_proba, 
        'hit': hit,
        'profit_to_loss': profit_to_loss,
        'consistency': consistency,
    }).round(2)
