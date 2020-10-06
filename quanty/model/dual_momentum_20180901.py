import numpy as np
import pandas as pd
from IPython.core.debugger import set_trace
from pandas.tseries.offsets import Day
from numba import njit, float64, int64, int32, boolean
import time



class DualMomentum(object):
    def __init__(self, **params):
        self.__dict__.update(**params)
        
        self.assets_score, self.assets_sig = self._assets()
        self.sig, self.sig_w = self._signal()
        self.has_trend, self.has_trend_sp, self.has_trend_market = self._trend()
        self.score, self.ranks = self._score()
        self.selection = self._selection_all()
        
        
    def _assets(self):
        assets_score = self.assets
        assets_sig = assets_score | {self.cash_equiv, self.supporter}
        if self.market is not None: assets_sig.update({self.market})

        return list(assets_score), list(assets_sig)
        
        
    def _trend(self):
        has_trend = self._has_trend(self.follow_trend).reindex(index=self.dates_asof, method='ffill') #.loc[self.dates_asof]
        has_trend_sp = self._has_trend(self.follow_trend_supporter, asset=self.supporter).reindex(index=self.dates_asof, method='ffill') 
        #.loc[self.dates_asof]
        
        has_trend_market = None
        if self.market is not None:
            has_trend_market = self._has_trend(self.follow_trend_market, asset=self.market).reindex(index=self.dates_asof, method='ffill') 
            #.loc[self.dates_asof]
            
        return has_trend, has_trend_sp, has_trend_market
    

    def _score(self):
        #score = self.sig[self.assets_score].copy() 
        #이렇게하면 assets_score에 supporter가 없는 경우, (즉 supporter가 assets_sig 에만 있는 경우)
        #supporter의 랭크가 ranks에서 빠져서, supporter가 weights에서 제외되므로, 
        #반드시 assets_score에 supporter를 추가해야한다. 
        
        score = self.sig.copy()
        score[list(set(score.columns)-set(self.assets_score))] = np.nan
        #set_trace()
        score[~self.has_trend] = np.nan
        ranks = score.rank(axis=1, ascending=False, na_option='bottom')
        return score, ranks
        
        
    #def _bet_of(self, asset):
    #    if asset in dict(self.overwrite_to_bet):
    #        return dict(self.overwrite_to_bet)[asset]
    #    else:
    #        return asset


    def _sig_dynamic_mix_by_n_fwd(self, n_fwd):
        n_backs = range(21*self.sig_dyn_m_backs, 0, -21)
        n_sample = self.sig_dyn_n_sample
        n_delay = 0
        pr = self.p_close[list(self.assets)]
        
        def _get_cor(n_back):
            #if n_back==441: set_trace()
            p1 = pr.shift(n_fwd+n_delay)
            p2 = pr
            perf_past = p1.pct_change(n_back)# / p1.pct_change().rolling(n_back).std()
            perf_fut = p2.pct_change(n_fwd)# / p2.pct_change().rolling(n_fwd).std()
            return perf_past.corrwith(perf_fut, axis=1).rolling(n_sample, min_periods=2).mean()
        
        return pd.DataFrame({n_back:_get_cor(n_back) for n_back in n_backs})[list(n_backs)].fillna(0)


    def _sig_dynamic_mix(self): 
        out = pd.DataFrame()
        for n_fwd in self.sig_dyn_fwd:
            out = out.add(self._sig_dynamic_mix_by_n_fwd(n_fwd)/n_fwd, fill_value=0)

        out /= sum(1/np.array(self.sig_dyn_fwd))
        out = out[(out>self.sig_dyn_thres) | (out<-self.sig_dyn_thres)]        
        return out.fillna(0)
                
        
    def _signal_with(self, sig_w):
        sig = []
        n_sig = sig_w.shape[1]
        pr = self.p_close[self.assets_sig].resample('M').ffill()
        pr.index = self.p_close.index[self.p_close.index.get_indexer(pr.index, method='ffill')]
        
        def __sig_at(date):
            #sig_w_ = sig_w.loc[:date].iloc[-1]
            sig_w_ = sig_w.reindex(index=[date], method='ffill').iloc[0]
            #set_trace()
            pr_ = pr.loc[:date].iloc[-n_sig-1:]
            rt = (pr_.iloc[-1]/pr_.iloc[:-1]-1).replace(np.inf, np.nan)
            sig_w_ = sig_w_.iloc[-len(rt):]
            return rt.mul(sig_w_.values, axis=0).sum(skipna=False)

        return pd.DataFrame([__sig_at(date) for date in self.dates_asof], index=self.dates_asof)
        
    
    
    def _sig_w(self):
        sig_w = self.sig_w_base
        
        if self.sig_w_dynamic:
            #st = time.time()
            mixer = self._sig_dynamic_mix()
            #print(time.time()-st)
            sig_w_ = np.zeros(mixer.shape[1])
            sig_w_[-len(sig_w):] = sig_w
            return mixer.add(sig_w_)
        
        else:
            return pd.DataFrame([sig_w]*len(self.dates_asof), index=self.dates_asof)
    
        
                
    def _signal(self):
        sig_w = self._sig_w()
        sig = self._signal_with(sig_w)
        return sig, sig_w


    def _selection_all(self):
        return pd.DataFrame([self._selection(date) for date in self.dates_asof], index=self.dates_asof)   


    def _selection(self, date):
        pos_sp = pos_cash = 0

        sp_has_trend = self.has_trend_sp.loc[date]
        sp_has_positive_sig = self.sig.loc[date, self.supporter]>=0
        cash_has_positive_sig = self.sig.loc[date, self.cash_equiv]>=0
        
        if self.support_cash and (sp_has_trend or sp_has_positive_sig):
            pos = self._get_default_selection(date, self.n_picks-1)
            pos_sp = self.n_picks - pos.sum()

            if sp_has_trend and sp_has_positive_sig:
                pass

            elif sp_has_trend:
                pos_sp = int(pos_sp*0.5)

            elif sp_has_positive_sig:
                pos_sp = int(pos_sp*0.5)

        else:
            pos = self._get_default_selection(date, self.n_picks)

            if cash_has_positive_sig:# and cash_has_trend:
                pos_cash = self.n_picks - pos.sum()

        pos = self._selection_add(pos, self.supporter, pos_sp)
        pos = self._selection_add(pos, self.cash_equiv, pos_cash)
        
        return pos
    
    
    def _selection_add(self, pos, asset, value):
        try:
            pos.loc[asset] += value

        except:
            pos.loc[asset] = value
            
        return pos
    
    
    def _get_oversold(self, date):
        p_ = self.p_close.loc[:date].iloc[-20:]
        z = -((p_-p_.mean())/p_.std()).iloc[-1]
        return z.rank(ascending=False)<=3
            

    def _get_default_selection(self, date, n_picks):
        score = self.score.loc[date]
        ranks = self.ranks.loc[date]
        sig = self.sig.loc[date]
        
        if self.mode=='DualMomentum':
            pos = (score>0) & (ranks<=n_picks)
            if self.market is not None:
                #pos &= (sig.loc[self.market]>0)
                pos &= (self.has_trend_market[date]) | (sig.loc[self.market]>0)# | (ranks<=ranks[self.market])
                #|(sig.loc[self.market]>sig.loc[self.supporter])
                #pos &= (sig.loc[self.market]>=sig.loc[self.supporter])
                #pos &= (ranks<ranks[self.market])
          
        elif self.mode=='RelativeMomentum':
            pos = ranks<=n_picks
          
        elif self.mode=='AbsoluteMomentum':
            pos = (score>0)
            if self.market is not None:
                pos &= (sig.loc[self.market]>0)
                
        #elif self.mode=='Rebound':
        #    p_ = self.p_close.loc[:date].iloc[-20:]
        #    z = -((p_-p_.mean())/p_.std()).iloc[-1]
        #    pos = z.rank(ascending=False)<=n_picks
                
        #if sum(pos)<=0:
            #set_trace()
            #pos |= (self._get_oversold(date) & (sig>0))
        
        return pos.astype(int)

    
    def _has_trend(self, terms, asset=None):
        if asset is None:
            ma_short = self.p_close.rolling(terms[0]).mean()#, min_periods=2).mean() #
            ma_long = self.p_close.rolling(terms[1]).mean()#, min_periods=2).mean()
            
        else:
            ma_short = self.p_close[asset].rolling(terms[0]).mean()#, min_periods=2).mean()
            ma_long = self.p_close[asset].rolling(terms[1]).mean()#, min_periods=2).mean()
            
        return ma_short>ma_long    
    
    