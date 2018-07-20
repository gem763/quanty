import numpy as np
import pandas as pd
from IPython.core.debugger import set_trace
from pandas.tseries.offsets import Day
from numba import njit, float64, int64, int32, boolean


@njit#(float64[:](int64, int64[:], float64[:,:], int32[:,:]))
def _signal_nb(i_date, i_ref, p_ref_val, sig_w):
    r = p_ref_val[i_date] / p_ref_val[i_ref[i_ref<i_date][-len(sig_w):]] - 1.0
    r *= sig_w[-r.shape[0]:]
    
    n = p_ref_val.shape[1]
    out = np.empty(n)
    
    for col in range(n):
        out[col] = r[:,col].sum()
    
    return out
    

@njit#(float64[:,:](int64[:], int64[:], float64[:,:], int32[:,:]))
def _signal_all_nb(i_dates, i_ref, p_ref_val, sig_w):
    out = np.empty((len(i_dates), p_ref_val.shape[1]))
        
    for i,i_date in enumerate(i_dates):
        out[i,:] = _signal_nb(i_date, i_ref, p_ref_val, sig_w)
        
    return out
    
    
@njit#(boolean(int64, int64, int64, float64[:,:], int64))
def _has_ma_mtum_single_nb(i_date, term_short, term_long, p_ref_val, i_asset):
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
        #self.p_close_val = self.p_close.values
        self.sig_w = np.array(self.sig_w).reshape(-1,1)
        
        self.sig, self.is_tradable = self._signal()
        self.selection, self.ranks = self._selection_all()

        
    def _bet_of(self, asset):
        if asset in dict(self.overwrite_to_bet):
            return dict(self.overwrite_to_bet)[asset]
        else:
            return asset

        
    def _signal(self):
        sig = _signal_all_nb(self.i_dates, self.i_ref, self.p_ref_val, self.sig_w)
        sig = pd.DataFrame(sig, index=self.dates_asof, columns=self.assets).rename(columns=dict(self.overwrite_to_bet))
        is_tradable = self.p_close.reindex(index=self.dates_asof, method='ffill').notnull()
        sig[~is_tradable] = np.nan

        return sig, is_tradable


    def _selection_all(self):
        selection = []
        ranks = []
        
        for i, i_date in enumerate(self.i_dates):
            selection_, ranks_ = self._selection(self.sig.iloc[i], self.is_tradable.iloc[i], i_date)
            selection.append(selection_)
            ranks.append(ranks_)
        
        selection = pd.DataFrame(selection, index=self.dates_asof)
        ranks = pd.DataFrame(ranks, index=self.dates_asof)
        return selection, ranks
    
    
    def _selection(self, sig, is_tradable, i_date):
        has_rf_ma_mtum = self._has_ma_mtum_single(i_date, self.follow_trend_riskfree, self.riskfree)
        #set_trace()
        has_rf_positive_sig = sig.loc[self._bet_of(self.riskfree)]>=0
        
        if self.support_cash and is_tradable[self._bet_of(self.riskfree)] and (has_rf_ma_mtum or has_rf_positive_sig):
            pos, ranks = self._get_default_selection(i_date, sig, self.n_picks-1)
            pos_rf = self.n_picks - pos.sum()
            pos_cash = 0
            
            if has_rf_ma_mtum and has_rf_positive_sig:
                pass
            
            elif has_rf_ma_mtum:
                pos_rf = int(pos_rf*0.5)
              
            elif has_rf_positive_sig:
                pos_rf = int(pos_rf*0.5)
          
        else:
            pos, ranks = self._get_default_selection(i_date, sig, self.n_picks)
            pos_rf = 0

            if is_tradable[self._bet_of(self.cash_equiv)]:
                pos_cash = self.n_picks - pos.sum()
                
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
        if terms is not None:
            i_asset = self.p_ref.columns.get_loc(asset_ref)
            has_ma_mtum = _has_ma_mtum_single_nb(i_date, terms[0], terms[1], self.p_ref_val, i_asset)
        
        else:
            has_ma_mtum = True
            
        return has_ma_mtum    
    
