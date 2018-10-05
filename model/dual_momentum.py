import numpy as np
import pandas as pd
import time
from IPython.core.debugger import set_trace
from pandas.tseries.offsets import Day
from .portfolio import Port



class DualMomentumPort(Port):
    
    def __init__(self, dates_port, wealth=None, model_rtn=None, **params):
        self.selector = DualMomentumSelector(dates_port, **params)
        Port.__init__(self, dates_port, wealth=None, model_rtn=None, **params)
        
    
    def _get_pos(self, date):
        selection = self.selector.selection.loc[date]
        sig = self.selector.sig.loc[date]
        ranks = self.selector.ranks.loc[date]

        if self.w_type=='ew':
            pos = selection

        elif self.w_type=='ranky':
            pos = selection / ranks

        elif self.w_type=='ranky2':
            pos = selection / (ranks**0.5)
            
        elif self.w_type=='inv_ranky':
            pos = selection * ranks

        elif self.w_type=='inv_ranky2':
            pos = selection * (ranks**0.5)
            
        elif self.w_type=='sig':
            pos = self._get_post_sig(selection, sig)
            
        elif self.w_type=='iv':
            pos = self._get_pos_iv(selection, date)
    
        elif self.w_type=='eaa':
            pos = self._get_pos_eaa(sig, date)
            
        elif self.w_type=='eaa_mod':
            pos = self._get_pos_eaa_mod(selection, sig, date)
            
        elif self.w_type=='eaa_optima': 
            pos = self._get_pos_eaa_optima(selection, sig, date)

        # Normalize
        pos /= pos.sum()
        return pos.fillna(0)

    
    
    def _get_pos_sig(self, selection, sig):
        sig_ = sig[selection!=0]
        sig_[sig_<=0] = sig_[sig_>0].mean()
        return selection * sig_

    
    def _get_pos_iv(self, selection, date):
        i_date = self.p_close.index.get_loc(date, method='ffill')
        df = self.p_close.iloc[i_date+1-self.iv_period:i_date+1]

        # Underlying index 중에 종종 일일데이터가 없는 경우가 있다
        # 이 종목은 변동성이 매우 작을 것이므로, 제거한다
        has_enough = pd.Series({k:df[k].nunique() for k in df}) > self.iv_period/2.0
        std = df[has_enough.index[has_enough]].pct_change().std()
        return selection / std
    
    
    def _get_pos_eaa(self, sig, date):
        pr = self.p_close
        rt = pr.loc[pr.index<date, sig.index].iloc[-251:].pct_change().iloc[1:]
        #rt_short = rt.iloc[-20:]
        rt_ew = rt.mean(axis=1)
        cor = rt.corrwith(rt_ew)
        #cor = rt_short.cov().dot(np.ones(len(sig)))/rt_short.std()/((rt_short.cov().sum().sum())**0.5)

        sig_ = sig.copy()
        sig_[sig_<0] = 0

        eaa_wr = 1e-6 if self.eaa_wr==0 else self.eaa_wr
        score = (sig_**eaa_wr) * ((1-cor)**self.eaa_wc)
        ranks = score.rank(ascending=False, na_option='bottom')
        sel = (sig_>0) & (score>0) & (ranks<1+self.n_picks)
        sel = sel.astype(int)
        sel.loc[self.cash_equiv] += (self.n_picks - sel.sum())
        #sel.loc[self._bet_of(self.cash_equiv)] += (self.n_picks - sel.sum())
        return sel*score
    
    
    def _get_pos_eaa_mod(self, selection, sig, date):
        pr = self.p_close
        rt = pr.loc[pr.index<date, selection.index].iloc[-251:].pct_change().iloc[1:]
        rt_ew = rt.mean(axis=1)
        cor = rt.corrwith(rt_ew)
        sig_ = sig[selection!=0]
        sig_[~(sig_>0)] = sig_[sig_>0].mean()
        return selection * ((sig_**self.eaa_wr) * ((1-cor)**self.eaa_wc))
    
    
    def _get_pos_eaa_optima(self, selection, sig, date):
        pr = self.p_close
        rt = pr.loc[pr.index<date, selection.index].iloc[-251:].pct_change().iloc[1:]
        rt_ew = rt.mean(axis=1)
        cor = rt.corrwith(rt_ew)

        n_grid = 2 * self.eaa_wr_bnd + 1
        wrs = np.linspace(-self.eaa_wr_bnd, self.eaa_wr_bnd, n_grid)
        wcs = np.array([self.eaa_wc])
        rt_short = rt.iloc[-self.eaa_short_period:]

        sig_ = sig[selection!=0]
        #sig_[~(sig_>0)] = sig_[sig_>0].mean()
        #cor = rt_short.cov().dot(np.ones(len(selection)))/rt_short.std()/((rt_short.cov().sum().sum())**0.5)

        def score_(wr_, wc_):
            pos_ = selection * ((sig_**wr_) * ((1-cor)**wc_))
            pos_.fillna(0, inplace=True)
            #pos_ /= pos_.sum() 
            #rt_expected = (sig_ * pos_).sum()
            rt_expected = (rt_short.sum() * pos_).sum()
            #vol = (pos_.T.dot(rt.cov()).dot(pos_))**0.5
            return rt_expected# / vol#- (pos_**2).sum()

        candidates = np.array([[score_(wr_, wc_) for wc_ in wcs] for wr_ in wrs])
        i_wr, i_wc = np.unravel_index(candidates.argmax(), candidates.shape)
        wr = wrs[i_wr]
        wc = wcs[i_wc]
        self.wr.append(wr)
        self.wc.append(wc)

        return selection * ((sig_**wr) * ((1-cor)**wc))
    
        
        

class DualMomentumSelector(object):
    def __init__(self, dates_port, **params):
        dates_port = pd.DatetimeIndex(dates_port)
        
        self.__dict__.update(**params)
        self.assets_score, self.assets_sig = self._assets()
        
        self.sig, self.sig_w = self._signal(dates_port)
        self.has_trend, self.has_trend_sp, self.has_trend_market = self._trend(dates_port)
        self.score, self.ranks = self._score()
        self.selection = self._selection()

        
    def _assets(self):
        assets_score = self.assets #| {self.supporter}
        assets_sig = assets_score | {self.cash_equiv, self.supporter}
        if self.market is not None: assets_sig.update({self.market})

        return list(assets_score), list(assets_sig)
        
        
    def _score(self):    
        #score = self.sig[self.assets_score].copy() 
        #이렇게하면 assets_score에 supporter가 없는 경우, (즉 supporter가 assets_sig 에만 있는 경우)
        #supporter의 랭크가 ranks에서 빠져서, supporter가 weights에서 제외되므로, 
        #반드시 assets_score에 supporter를 추가해야한다. 
        #set_trace()
        score = self.sig.copy()
        score[list(set(score.columns)-set(self.assets_score))] = np.nan
        
        if self.has_trend is not None:
            score[~self.has_trend] = np.nan
            
        if self.has_trend_market is not None:
            score[(self.sig[self.market]<=0) & (~self.has_trend_market)] = np.nan
            
        elif self.market is not None:
            score[self.sig[self.market]<=0] = np.nan
            
        ranks = score.rank(axis=1, ascending=False, na_option='bottom')        
        return score, ranks                    
        
        
    def _trend(self, dates):
        has_trend = has_trend_sp = has_trend_market = None
        
        if self.follow_trend is not None:
            has_trend = self._has_trend(dates, self.follow_trend)
            
        if (self.follow_trend_supporter is not None) & (self.supporter is not None):
            has_trend_sp = self._has_trend(dates, self.follow_trend_supporter, asset=self.supporter)
        
        if (self.follow_trend_market is not None) & (self.market is not None):
            has_trend_market = self._has_trend(dates, self.follow_trend_market, asset=self.market)

        return has_trend, has_trend_sp, has_trend_market

    
    def _has_trend(self, dates, terms, asset=None):
        if asset is None:
            pr = self.p_close.loc[:dates[-1], self.assets_sig]

        else:
            pr = self.p_close.loc[:dates[-1], asset]

        ma_short = pr.rolling(terms[0]).mean()
        ma_long = pr.rolling(terms[1]).mean()

        has_trend = ma_short>=ma_long
        return has_trend.reindex(dates, method='ffill')
        
        
        
    def _signal_with(self, sig_w, dates):    
        n_sig = sig_w.shape[1]
                
        def __sig_at(date):
            sig_w_ = sig_w.loc[date]            
            pr_ = self.p_close.loc[:date, self.assets_sig]
            pr__ = pr_[::-self.sig_w_term][:n_sig+1][::-1]
            rt = (pr__.iloc[-1]/pr__.iloc[:-1]-1).replace(np.inf, np.nan)
            #std = pd.DataFrame([pr_.iloc[-i_sig*self.sig_w_term-1:].pct_change().std() for i_sig in range(n_sig,0,-1)], index=rt.index)
            
            sig_w_ = sig_w_.iloc[-len(rt):]#; set_trace()
            return (rt).mul(sig_w_.values, axis=0).sum(skipna=False)
        
        return pd.DataFrame([__sig_at(date) for date in dates], index=dates)

    
    def _signal_with2(self, sig_w, dates):    
        n_sig = sig_w.shape[1]
                
        def __sig_at(date):
            sig_w_ = sig_w.xs(date, level=0)
            pr_ = self.p_close.loc[:date, self.assets_sig]
            pr__ = pr_[::-self.sig_w_term][:n_sig+1][::-1]
            rt = (pr__.iloc[-1]/pr__.iloc[:-1]-1).replace(np.inf, np.nan)
            #std = pd.DataFrame([pr_.iloc[-i_sig*self.sig_w_term-1:].pct_change().std() for i_sig in range(n_sig,0,-1)], index=rt.index)
            
            #set_trace()
            #sig_w_ = sig_w_.iloc[-len(rt):]
            rt.index = sig_w_.columns[-len(rt):]
            #std.index = sig_w_.columns[-len(rt):]
            return (rt).mul(sig_w_.T).sum(skipna=False)
            #return (rt).mul(sig_w_.values, axis=0).sum(skipna=False)
        
        return pd.DataFrame([__sig_at(date) for date in dates], index=dates)    
    
            
    def _sig_w(self, dates):
        sig_w_base = [0]*12 if self.sig_w_base is None else self.sig_w_base
        #set_trace()
        if self.sig_w_dynamic:
            mixer = self._sig_dynamic_mix(dates)
            
            if mixer.shape[1]>len(sig_w_base): 
                sig_w_ = np.zeros(mixer.shape[1])
                sig_w_[-len(sig_w_base):] = sig_w_base
                
            else:
                sig_w_ = sig_w_base[-mixer.shape[1]:]
                
            return mixer.add(sig_w_)
        
        else:
            return pd.DataFrame([sig_w_base]*len(dates), columns=range(self.sig_w_term*len(sig_w_base), 0, -self.sig_w_term), index=dates)
    
              
    def _signal(self, dates):
        sig_w = self._sig_w(dates)
        sig = self._signal_with(sig_w, dates)
        return sig, sig_w
    
    
    def _sig_dynamic_mix_by_n_fwd(self, dates, pr, n_backs, n_fwd):
        n_delay = 0

        pr_ = pr
        p1 = pr_.shift(n_fwd+n_delay)
        p2 = pr_
        perf_fut_rt = p2.pct_change(n_fwd)
        perf_fut_std = p2.pct_change().rolling(n_fwd).std()
        perf_fut = perf_fut_rt/perf_fut_std
        
        def _get_cor(n_back):
            perf_past_rt = p1.pct_change(n_back)
            perf_past_std = p1.pct_change().rolling(n_back).std()
            perf_past = perf_past_rt/perf_past_std
            cor = perf_past_rt.corrwith(perf_fut_rt, axis=1).rolling(self.sig_dyn_n_sample, min_periods=2).mean()
            #set_trace()
            return cor
            #return perf_past.corrwith(perf_fut, axis=1).rolling(self.sig_dyn_n_sample, min_periods=2).mean()
            #return perf_past.corrwith(perf_fut, axis=1).ewm(halflife=250).mean()

        mixer = pd.DataFrame({n_back:_get_cor(n_back) for n_back in n_backs})[n_backs]#.fillna(0)
        mixer = mixer.reindex(index=dates, method='ffill').fillna(0)
        mixer[(mixer<=self.sig_dyn_thres) & (mixer>=-self.sig_dyn_thres)] = 0
        return mixer
        

    def _sig_dynamic_mix_by_n_fwd2(self, dates, pr, n_backs, n_fwd):
        n_delay = 0

        pr_ = pr
        p1 = pr_.shift(n_fwd+n_delay)
        p2 = pr_
        perf_fut_rt = p2.pct_change(n_fwd)
        perf_fut_std = p2.pct_change().rolling(n_fwd).std()
        perf_fut = perf_fut_rt/perf_fut_std
        
        def _get_cor(n_back):
            perf_past_rt = p1.pct_change(n_back)
            perf_past_std = p1.pct_change().rolling(n_back).std()
            perf_past = perf_past_rt/perf_past_std
            #set_trace()
            #cor = perf_past_rt.corrwith(perf_fut_rt, axis=1).rolling(self.sig_dyn_n_sample, min_periods=2).mean()
            #set_trace()
            #perf_past_rt.loc[:dates[0]].iloc[::-5].iloc[:20]
            #cor = pd.DataFrame([perf_past_rt.loc[:date].iloc[-20:].corrwith(perf_fut_rt.loc[:date].iloc[-20:]) for date in dates], index=dates)
            cor = pd.DataFrame([
                perf_past_rt.loc[:date].iloc[::-5].iloc[:50].corrwith(perf_fut_rt.loc[:date].iloc[::-5].iloc[:50])
                for date in dates
            ], index=dates)
            #set_trace()
            return cor.stack()
            #return perf_past.corrwith(perf_fut, axis=1).rolling(self.sig_dyn_n_sample, min_periods=2).mean()
            #return perf_past.corrwith(perf_fut, axis=1).ewm(halflife=250).mean()

        
        mixer = pd.DataFrame({n_back:_get_cor(n_back) for n_back in n_backs})[n_backs]#.fillna(0)
        #set_trace()
        #mixer = mixer.reindex(index=dates, method='ffill').fillna(0)
        mixer[(mixer<=self.sig_dyn_thres) & (mixer>=-self.sig_dyn_thres)] = 0
        return mixer        
        
        

    def _sig_dynamic_mix2(self, dates):
        n_backs = list(range(self.sig_w_term*self.sig_dyn_m_backs, 0, -self.sig_w_term))
        pr = self.p_close.loc[:dates[-1], self.assets_score]
        #out = pd.DataFrame()
        
        div = []
        for i_fwd, n_fwd in enumerate(self.sig_dyn_fwd):
            #set_trace()
            if i_fwd==0:
                out = self._sig_dynamic_mix_by_n_fwd(dates, pr, n_backs, n_fwd)/(i_fwd+1)
            else:
                out = out.add(self._sig_dynamic_mix_by_n_fwd2(dates, pr, n_backs, n_fwd)/(i_fwd+1), fill_value=0)
            #out = out.add(self._sig_dynamic_mix_by_n_fwd(dates, pr, n_backs, n_fwd)/(i_fwd+1), fill_value=0)
            div.append(i_fwd+1)
            

        #set_trace()
        out /= sum(1/np.array(div))
        return out        
 

    def _sig_dynamic_mix(self, dates):
        n_backs = list(range(self.sig_w_term*self.sig_dyn_m_backs, 0, -self.sig_w_term))
        pr = self.p_close.loc[:dates[-1], self.assets_score]
        out = pd.DataFrame()
        
        div = []
        for i_fwd, n_fwd in enumerate(self.sig_dyn_fwd):
            out = out.add(self._sig_dynamic_mix_by_n_fwd(dates, pr, n_backs, n_fwd)/(i_fwd+1), fill_value=0)
            div.append(i_fwd+1)
            
        out /= sum(1/np.array(div))
        return out        

        
    def _n_picks(self):
        if isinstance(self.n_picks, int):
            return self.n_picks
        
        else:
            return int(len(self.assets_score) * self.n_picks)
            #return int(self.p_close[self.assets_score].loc[:date].iloc[-1].count() * self.n_picks)
        

    def _selection(self):
        def __selection(date):
            pos_sp = pos_cash = 0
            n_picks = self._n_picks()
            
            if self.strong_condition:
                sp_has_trend = self.has_trend_sp.loc[date] if self.has_trend_sp is not None else False
                sp_has_positive_sig = self.sig.loc[date, self.supporter]>=0
                
                if sp_has_trend or sp_has_positive_sig:
                    pos = self._get_default_selection(date, n_picks)#-1)
                    pos_sp = n_picks - pos.sum()

                    if sp_has_trend and sp_has_positive_sig:
                        pass

                    elif sp_has_trend:
                        pass
                        #pos_sp = pos_sp*0.5

                    elif sp_has_positive_sig:
                        pass
                        #pos_sp = pos_sp*0.5

                else:
                    cash_has_positive_sig = self.sig.loc[date, self.cash_equiv]>=0
                    pos = self._get_default_selection(date, n_picks)

                    if cash_has_positive_sig:# and cash_has_trend:
                        pos_cash = n_picks - pos.sum()
            
            else:
                pos = self._get_default_selection(date, n_picks)
                pos_sp = n_picks - pos.sum()
            
            
            pos = self._selection_add(pos, self.supporter, pos_sp)
            pos = self._selection_add(pos, self.cash_equiv, pos_cash)

            return pos
        
        return pd.DataFrame([__selection(date) for date in self.sig.index], index=self.sig.index)
    
    
    
    def _selection_add(self, pos, asset, value):
        try:
            pos.loc[asset] += value

        except:
            pos.loc[asset] = value
            
        return pos
    
    
#    def _get_oversold(self, date):
#        p_ = self.p_close.loc[:date].iloc[-20:]
#        z = -((p_-p_.mean())/p_.std()).iloc[-1]
#        return z.rank(ascending=False)<=3
            

    def _get_default_selection(self, date, n_picks):
        score = self.score.loc[date]
        ranks = self.ranks.loc[date]
        sig = self.sig.loc[date]
               
        #if date==pd.Timestamp('2008-09-30'): set_trace()
        if self.mode=='DualMomentum':
            pos = (score>0) & (ranks<=n_picks)
            #if self.market is not None:
                # self.has_trend_market[date]는 Supporter에 몰빵하는 것을 막기위한 장치
            #    pos &= (sig[self.market]>0) | (self.has_trend_market[date])
          
        elif self.mode=='RelativeMomentum':
            pos = ranks<=n_picks
          
        elif self.mode=='AbsoluteMomentum':
            pos = (score>0)
            #if self.market is not None:
            #    pos &= (sig[self.market]>0)
                        
        return pos.astype(int)        
        
        