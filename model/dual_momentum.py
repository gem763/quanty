import numpy as np
import pandas as pd
from IPython.core.debugger import set_trace
from pandas.tseries.offsets import Day


class DualMomentum(object):
    
    def __init__(self, **params):
        #mode
        #n_picks
        #sig_w
        #p_ref
        #p_close
        #assets_member_bet
        #cash_equiv
        #riskfree
        #market
        #rf_trend
        #support_cash
        #overall_market_check
        self.__dict__.update(**params)


    def get(self, date):
        sig_ = self._signal(date)
        selection_, ranks_ = self._selection(sig_, date)
        return selection_, ranks_, sig_
    
    
    def _signal(self, date):
        n_sig_w = len(self.sig_w)
        n_back = n_sig_w*31 + 40
        date_from = date - n_back*Day()
        date_to = date - 0*Day()
        p = self.p_ref.loc[date_from:date_to].resample('M').ffill().iloc[-n_sig_w-1:]
        r = (p.iloc[-1]/p.iloc[:-1]-1).replace(np.inf, np.nan)
        sig_w = self.sig_w[-len(r):]
        
        sig = r.mul(sig_w, axis=0).sum(skipna=False)        
        sig.index = self.assets_member.bet
        
        not_tradable = ~self._is_tradable(date)
        sig.loc[not_tradable] = np.nan
        return sig
    
    
    def _is_tradable(self, date, asset=None):
        if type(asset) is str:
            return not np.isnan(self.p_close.loc[:date, asset].iloc[-1])
          
        else:
            return self.p_close.loc[:date].iloc[-1].notnull()
            
    
    def _selection(self, sig, date):
        is_rf_tradable = self._is_tradable(date, self.riskfree)
        has_rf_ma_mtum = self._has_ma_mtum_single(date, self.rf_trend, self.riskfree)
        has_rf_positive_sig = sig.loc[self.riskfree]>=0
        
        if self.support_cash and is_rf_tradable and (has_rf_ma_mtum or has_rf_positive_sig):
            pos, ranks = self._get_default_selection(date, sig, self.n_picks-1)
            pos_rf = self.n_picks - pos.sum()
            pos_cash = 0
            
            if has_rf_ma_mtum and has_rf_positive_sig:
                pass
            
            elif has_rf_ma_mtum:
                pos_rf = int(pos_rf*0.5)
              
            elif has_rf_positive_sig:
                pos_rf = int(pos_rf*0.5)
          
        else:
            pos, ranks = self._get_default_selection(date, sig, self.n_picks)
            pos_rf = 0
            
            if self._is_tradable(date, self.cash_equiv):
                pos_cash = self.n_picks - pos.sum()
                
            else:
                pos_cash = 0
                
          
        pos.loc[self.riskfree] += pos_rf  
        
        try:
            pos.loc[self.cash_equiv] += pos_cash
            
        except:
            pos.loc[self.cash_equiv] = pos_cash
        
        return pos, ranks
    
    
    def _get_default_selection(self, date, sig, n_picks):
        has_ma_mtum = self._has_ma_mtum(date, self.self_trend)
        
        score = sig.copy()
        score.loc[~has_ma_mtum] = np.nan
        ranks = score.rank(ascending=False, na_option='bottom')
        
        if self.mode=='DualMomentum':
            pos = (score>0) & (ranks<1+n_picks)
            if self.overall_market_check:
                pos &= (sig.loc[self.market]>0)
          
        elif self.mode=='RelativeMomentum':
            pos = ranks<1+n_picks
          
        elif self.mode=='AbsoluteMomentum':
            pos = (score>0)
            if self.overall_market_check:
                pos &= (sig.loc[self.market]>0)
                
        return pos.astype(int), ranks
    
    
    
    def _has_ma_mtum(self, date, terms):
        if terms is not None:
            p = self.p_ref.loc[:date]
            p_ma_short = p.iloc[-terms[0]:].mean()
            p_ma_long = p.iloc[-terms[1]:].mean()
            has_ma_mtum = p_ma_short > p_ma_long
            has_ma_mtum.index = self.assets_member.bet
        
        else:
            has_ma_mtum = pd.Series(index=self.assets_member.bet)
            has_ma_mtum[:] = True
            
        return has_ma_mtum

      
    def _has_ma_mtum_single(self, date, terms, asset_ref):
        if terms is not None:
            p = self.p_ref.loc[:date, asset_ref]
            p_ma_short = p.iloc[-terms[0]:].mean()
            p_ma_long = p.iloc[-terms[1]:].mean()
            has_ma_mtum = p_ma_short > p_ma_long
        
        else:
            has_ma_mtum = True
            
        return has_ma_mtum
    