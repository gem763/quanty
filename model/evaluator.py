import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
from numba import jit, vectorize, int64, float64, void


# 종목별 CAGR
@jit(float64(float64[:]), nopython=True)
def _cagr(p):
    return ((p[-1]/p[0])**(250/(len(p)-1)) - 1) * 100


# 종목별 변동성
def _std2(p): 
    return np.nanstd(p[1:]/p[:-1]-1) * (250**0.5) * 100


@jit(float64(float64[:]), nopython=True)
def _std(p):
    rt = p[1:]/p[:-1] - 1.0
    n = len(rt)
    rt_mean = rt.sum() / n
    out = (((rt-rt_mean)**2).sum() / (n-1))**0.5
    return out * (250**0.5) * 100


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


# 종목별 Sharpe
@jit(float64(float64[:]), nopython=True)
def _sharpe(p): 
    return _cagr(p)/_std(p)


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


def _stats(cum, beta_to, n_roll_stats):
    #cum = self.cum
    cum_last = cum.iloc[-1]
    n_samples = cum.count()

    # Base stats: 전체구간
    cagr = (cum_last**(250/(n_samples-1)) - 1) * 100
    std = cum.pct_change().std() * (250**0.5) * 100
    sharpe = cagr / std
    #mdd = (cum.expanding().apply(lambda x: x[-1]/np.nanmax(x)-1).min()) * 100
    mdd = (cum.div(cum.cummax()) - 1).min() * 100
    #set_trace()
    consistency = (pd.Series([_consistency(cum[col].dropna()) for col in cum], index=cum.columns)) * 100
    beta = pd.Series([_beta(cum[[col, beta_to]].pct_change().dropna(how='any').values) for col in cum], index=cum.columns)

    # Rolling stats
    cum_roll = cum.rolling(n_roll_stats)
    cagr_roll = cum_roll.apply(_cagr)
    cagr_roll_med = cagr_roll.median()
    loss_proba = (cagr_roll<0).sum()/cagr_roll.count() * 100
    std_roll_med = cum_roll.apply(_std).median()
    sharpe_roll_med = cum_roll.apply(_sharpe).median()

    # With 1M returns
    r_month = cum.resample('M').ffill().pct_change()
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




'''
class Evaluator(object):
    
    # 종목별 CAGR
    @classmethod
    def _cagr(cls, p):
        return ((p[-1]/p[0])**(250/(len(p)-1)) - 1) * 100


    # 종목별 변동성
    @classmethod
    def _std(cls, p): 
        return np.nanstd(p[1:]/p[:-1]-1) * (250**0.5) * 100


    # 종목별 하방변동성
    @classmethod
    def _std_down(cls, p):
        r = p[1:]/p[:-1]-1
        r = r[~np.isnan(r)]
        r_neg = r[r<0]
        return ((((r_neg**2).sum() / (len(r)-1))**0.5) * (250**0.5)) * 100

      
    @classmethod
    def _std_dir_by_r(cls, r, dir_): 
        if dir_==1:
            r_dir = r[r>0]
        elif dir_==-1:    
            r_dir = r[r<0]
        return ((((r_dir**2).sum() / (len(r)-1))**0.5) * (250**0.5)) * 100

      
    @classmethod
    def _std_dir(cls, p, dir_): 
        r = p[1:]/p[:-1]-1
        r = r[~np.isnan(r)]
        if dir_==1:
            r_dir = r[r>0]
        elif dir_==-1:    
            r_dir = r[r<0]
        return ((((r_dir**2).sum() / (len(r)-1))**0.5) * (250**0.5)) * 100      
      

    # 종목별 Sharpe
    @classmethod
    def _sharpe(cls, p): 
        return cls._cagr(p)/cls._std(p)

    
    # 종목별 MDD
    @classmethod
    def _mdd(cls, p):
        return (p/np.maximum.accumulate(p)-1).min() * 100
        
  
    # 종목별 수익안정성
    @classmethod
    def _consistency(cls, p):
        if len(p)==0:
            return np.nan
          
        y = np.log(p)
        X = np.arange(len(p)).reshape(-1,1)
        model = linear_model.LinearRegression(fit_intercept=False).fit(X, y)
        return r2_score(y, model.predict(X))
  
  
    # 종목별 베타
    @classmethod
    def _beta(cls, rtns):
        if len(rtns)==0:
            return np.nan
          
        y = rtns[:,0].reshape(-1,1)
        X = rtns[:,1].reshape(-1,1)
        model = linear_model.LinearRegression().fit(X, y)
        return model.coef_[0,0]
  
  
    @classmethod
    def get_stats(cls, cum, beta_to, n_roll_stats):
        #cum = self.cum
        cum_last = cum.iloc[-1]
        n_samples = cum.count()
        
        # Base stats: 전체구간
        cagr = (cum_last**(250/(n_samples-1)) - 1) * 100
        std = cum.pct_change().std() * (250**0.5) * 100
        sharpe = cagr / std
        #mdd = (cum.expanding().apply(lambda x: x[-1]/np.nanmax(x)-1).min()) * 100
        mdd = (cum.div(cum.cummax()) - 1).min() * 100
        #set_trace()
        consistency = (pd.Series([cls._consistency(cum[col].dropna()) for col in cum], index=cum.columns)) * 100
        beta = pd.Series([cls._beta(cum[[col, beta_to]].pct_change().dropna(how='any').values) for col in cum], index=cum.columns)

        # Rolling stats
        cum_roll = cum.rolling(n_roll_stats)
        cagr_roll = cum_roll.apply(cls._cagr)
        cagr_roll_med = cagr_roll.median()
        loss_proba = (cagr_roll<0).sum()/cagr_roll.count() * 100
        std_roll_med = cum_roll.apply(cls._std).median()
        sharpe_roll_med = cum_roll.apply(cls._sharpe).median()

        # With 1M returns
        r_month = cum.resample('M').ffill().pct_change()
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
'''