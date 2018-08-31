import numpy as np
import pandas as pd
from IPython.core.debugger import set_trace
from pandas.tseries.offsets import Day
from numba import njit, float64, int64, int32, boolean


#@njit#(float64[:](int64, int64[:], float64[:,:], int32[:,:]))
def _signal_nb(i_date, i_ref, p_ref_val, sig_w):
    #sig_w += np.array([[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0]]).reshape(-1,1)
    #sig_w += np.array([[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0.25*4,0.25*6,0.25*12]]).reshape(-1,1)
    
    r = p_ref_val[i_date] / p_ref_val[i_ref[i_ref<i_date][-len(sig_w):]] - 1.0
    #r = p_ref_val[i_date] / p_ref_val[:i_date][::-21][::-1][-len(sig_w):] - 1.0
    
#    std_ = np.empty_like(r)
#    for ii in range(len(std_)):
#        pr = p_ref_val[i_date-(ii+1)*21:i_date]
#        rt = pr[1:]/pr[:-1]-1
#        std_[ii] = np.std(rt, axis=0)
#    std_ = std_[::-1]

    r *= sig_w[-r.shape[0]:]#*std_
    
    n = p_ref_val.shape[1]
    out = np.empty(n)
    
    for col in range(n):
        out[col] = r[:,col].sum()
    
    return out
    

#@njit#(float64[:,:](int64[:], int64[:], float64[:,:], int32[:,:]))
def _signal_all_nb(i_dates, i_ref, p_ref_val, sig_w):
    out = np.empty((len(i_dates), p_ref_val.shape[1]))
    
    for i,i_date in enumerate(i_dates):
        out[i,:] = _signal_nb(i_date, i_ref, p_ref_val, sig_w)
        #out[i,:] = _signal_nb(i_date, i_ref, p_ref_val, sig_w[i_date,:].reshape(-1,1))
      
    return out
    
    
@njit#(boolean(int64, int64, int64, float64[:,:], int64))
def _has_ma_mtum_single_nb(i_date, term_short, term_long, p_ref_val, i_asset):
    #set_trace()
    p = p_ref_val[:i_date+1,i_asset]
    p_ma_short = np.nanmean(p[-term_short:])
    p_ma_long = np.nanmean(p[-term_long:])
    return p_ma_short>p_ma_long
    

@njit#(boolean[:](int64, int64, int64, float64[:,:]))
def _has_ma_mtum_nb(i_date, term_short, term_long, p_ref_val):
    n = p_ref_val.shape[1]
    out = np.empty(n, dtype=boolean)
    
    for i_asset in range(n):
        out[i_asset] = _has_ma_mtum_single_nb(i_date, term_short, term_long, p_ref_val, i_asset)
        
    return out


@njit#(boolean[:,:](int64[:], int64, int64, float64[:,:]))
def _has_ma_mtum_all_nb(i_dates, term_short, term_long, p_ref_val):
    out = np.empty((len(i_dates), p_ref_val.shape[1]), dtype=boolean)
        
    for i,i_date in enumerate(i_dates):
        out[i,:] = _has_ma_mtum_nb(i_date, term_short, term_long, p_ref_val)
        
    return out


@njit#(boolean[:,:](int64[:], int64, int64, float64[:,:], int64))
def _has_ma_mtum_all_single_nb(i_dates, term_short, term_long, p_ref_val, i_asset):
    out = np.empty((len(i_dates), 1), dtype=boolean)

    for i,i_date in enumerate(i_dates):
        out[i,:] = _has_ma_mtum_single_nb(i_date, term_short, term_long, p_ref_val, i_asset)

    return out



class DualMomentum(object):
    def __init__(self, **params):
        self.__dict__.update(**params)
        
        dates_ref = pd.date_range(self.p_ref.index[0], self.p_ref.index[-1], freq='M')
        self.i_ref = self.p_ref.index.get_indexer(dates_ref, method='ffill')
        self.i_dates = self.p_ref.index.get_indexer(self.dates_asof, method='ffill')
        self.p_ref_val = self.p_ref.values
        #self.r_val = self.r.values
        self.sig_w = np.array(self.sig_w_base).reshape(-1,1)
        
        self.sig, self.is_tradable = self._signal()
        self.selection, self.ranks = self._selection_all()

        
    def _bet_of(self, asset):
        if asset in dict(self.overwrite_to_bet):
            return dict(self.overwrite_to_bet)[asset]
        else:
            return asset


    def _mom_mixer_by_n_fwd(self, n_fwd):    
        n_backs = range(504, 0, -21)
        n_sample = 60
        n_delay = 0

        pr = self.p_ref.copy().drop([self.riskfree], axis=1)
        
        #cov_ = lambda x,y: x.corrwith(y, axis=1)*x.std(axis=1)*y.std(axis=1)
        
        def _get_cor(p1, p2, n_back):
            perf_past = p1.pct_change(n_back)# / p1.pct_change().rolling(n_back).std()
            perf_fut = p2.pct_change(n_fwd)# / p2.pct_change().rolling(n_fwd).std()
            #set_trace()
            return perf_past.corrwith(perf_fut, axis=1).rolling(n_sample).mean() #.ewm(halflife=60).mean()
        
        mom_mixer = pd.DataFrame({
            #n_back:cov_(pr.shift(n_fwd+n_delay).pct_change(n_back), pr.pct_change(n_fwd)).rolling(n_sample).mean()
            n_back:_get_cor(pr.shift(n_fwd+n_delay), pr, n_back)
            for n_back in n_backs
        })[list(n_backs)]

        return mom_mixer.fillna(0)


    def _mom_mixer(self, n_fwds, thres=None): 
        out = pd.DataFrame()
        for n_fwd in n_fwds:
            out = out.add(self._mom_mixer_by_n_fwd(n_fwd)/n_fwd, fill_value=0)

        out /= sum(1/np.array(n_fwds))
        #out /= sum(n_fwds)
        #out /= len(n_fwds)
        out = out[(out>thres) | (out<-thres)]        
        return out.fillna(0)
        
        
    def _signal(self):
        #mom_mixer = self._mom_mixer([20,40,60], thres=0.1)
        #self.mom_mixer = mom_mixer
        
        sig = _signal_all_nb(self.i_dates, self.i_ref, self.p_ref_val, self.sig_w)
        #sig = _signal_all_nb(self.i_dates, self.i_ref, self.p_ref_val, mom_mixer.values)
        sig = pd.DataFrame(sig, index=self.dates_asof, columns=self.assets).rename(columns=dict(self.overwrite_to_bet))
        is_tradable = self.p_close.reindex(index=self.dates_asof, method='ffill').notnull()
        sig[~is_tradable] = np.nan

        return sig, is_tradable


    def _selection_all(self):
        selection = []
        ranks = []
        
        for i, i_date in enumerate(self.i_dates):
            selection_, ranks_ = self._selection(self.sig.iloc[i], self.is_tradable.iloc[i], i_date, self.n_picks)
            #selection_, ranks_ = self._selection_iter_n(self.sig.iloc[i], self.is_tradable.iloc[i], i_date)
            #self.sel_candidate.append(self._selection_iter_n(self.sig.iloc[i], self.is_tradable.iloc[i], i_date))
            selection.append(selection_)
            ranks.append(ranks_)
        
        selection = pd.DataFrame(selection, index=self.dates_asof)
        ranks = pd.DataFrame(ranks, index=self.dates_asof)#.drop([self.riskfree], axis=1)
        return selection, ranks
    

    def _selection_iter_n(self, sig, is_tradable, i_date):
        pr = self.p_close
        rt = pr.iloc[:i_date].iloc[-21:].pct_change().iloc[1:]
        rt_mean = rt.mean()
        cov = rt.cov()
        
        sel_candidate_ = []
        
        for n_ in self.n_picks_rng:
            selection_, ranks_ = self._selection(sig, is_tradable, i_date, n_)
            sel_ = selection_/selection_.sum()
            rt_exp = (sel_*rt_mean).sum()
            std = (sel_.T.dot(cov).dot(sel_))**0.5
            sel_candidate_.append(rt_exp/std)
    
        i_best = np.array(sel_candidate_).argmax()
        n_picks = self.n_picks_rng[i_best]
        
        return self._selection(sig, is_tradable, i_date, n_picks)
    
    
    def _selection(self, sig, is_tradable, i_date, n_picks):
        has_rf_ma_mtum = self._has_ma_mtum_single(i_date, self.follow_trend_riskfree, self.riskfree)
        has_rf_positive_sig = sig.loc[self._bet_of(self.riskfree)]>=0
        #set_trace()
        if self.support_cash and is_tradable[self._bet_of(self.riskfree)] and (has_rf_ma_mtum or has_rf_positive_sig):
            pos, ranks = self._get_default_selection(i_date, sig, n_picks-1)
            pos_rf = n_picks - pos.sum()
            pos_cash = 0
            
            if has_rf_ma_mtum and has_rf_positive_sig:
                pass
            
            elif has_rf_ma_mtum:
                pos_rf = int(pos_rf*0.5)
              
            elif has_rf_positive_sig:
                pos_rf = int(pos_rf*0.5)
          
        else:
            pos, ranks = self._get_default_selection(i_date, sig, n_picks)
            pos_rf = 0

            if is_tradable[self._bet_of(self.cash_equiv)]:
                pos_cash = n_picks - pos.sum()
                
            else:
                pos_cash = 0
                
          
        pos.loc[self._bet_of(self.riskfree)] += pos_rf  
        
        try:
            pos.loc[self._bet_of(self.cash_equiv)] += pos_cash
            
        except:
            pos.loc[self._bet_of(self.cash_equiv)] = pos_cash
        
        return pos, ranks
    
    
    def _get_default_selection(self, i_date, sig, n_picks):
        score = self._screen_by_ma_mtum(sig.copy(), i_date, self.follow_trend)
        ranks = score.rank(ascending=False, na_option='bottom')
        
        if self.mode=='DualMomentum':
            pos = (score>0) & (ranks<1+n_picks)
            if self.overall_market_check:
                pos &= (sig.loc[self._bet_of(self.market)]>0)
          
        elif self.mode=='RelativeMomentum':
            pos = ranks<1+n_picks
          
        elif self.mode=='AbsoluteMomentum':
            pos = (score>0)
            if self.overall_market_check:
                pos &= (sig.loc[self._bet_of(self.market)]>0)
                
        return pos.astype(int), ranks
    
    
    def _screen_by_ma_mtum(self, score, i_date, terms):
        if terms is not None:
            has_ma_mtum = _has_ma_mtum_nb(i_date, terms[0], terms[1], self.p_ref_val)
            score.loc[~has_ma_mtum] = np.nan
            
        return score
        
      
    def _has_ma_mtum_single(self, i_date, terms, asset_ref):
        #set_trace()
        if terms is not None:
            i_asset = self.p_ref.columns.get_loc(asset_ref)
            has_ma_mtum = _has_ma_mtum_single_nb(i_date, terms[0], terms[1], self.p_ref_val, i_asset)
        
        else:
            has_ma_mtum = True
            
        return has_ma_mtum    
    
