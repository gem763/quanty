import numpy as np
import pandas as pd
from pandas.tseries.offsets import Day
from ..model import evaluator as ev


class DualMomentum(object):
    
    def __init__(self, **params):
        #mode
        #dates
        #w_type
        #n_picks
        #sig_w
        #iv_period
        #p_ref
        #p_close
        #assets_member_bet
        #fill_cash
        #cash_equiv
        #riskfree
        #market
        #rf_trend
        #support_cash
        #overall_market_check
        #apply_kelly
        
        self.__dict__.update(**params)


    def get(self, date, wealth, model_rtn):
        sig_ = self._signal(date)
        weight_, pos_, ranks_, kelly_output = self._weights(sig_, date, wealth, model_rtn)
        return weight_, pos_, ranks_, kelly_output, sig_
    
    
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
        
        
    def _weights(self, sig, date, wealth, model_rtn):
        pos, ranks = self._get_selection(sig, date)
        
        if self.w_type=='ew':
            pos *= (1.0 / self.n_picks)

        elif self.w_type=='ranky':
            pos *= (1.0 / ranks)

        elif self.w_type=='ranky2':
            pos *= (1.0 / (ranks**0.5))
            
        elif self.w_type=='sig':
            sig_ = sig[pos!=0]
            #sig_plus = sig_[sig_>0]
            sig_[sig_<=0] = sig_[sig_>0].mean()
            pos *= sig_
            
        elif self.w_type=='iv':
            i_date = self.p_close.index.get_loc(date, method='ffill')
            df = self.p_close.iloc[i_date+1-self.iv_period:i_date+1]
            
            # Underlying index 중에 종종 일일데이터가 없는 경우가 있다
            # 이 종목은 변동성이 매우 작을 것이므로, 제거한다
            has_enough = pd.Series({k:df[k].nunique() for k in df}) > self.iv_period/2.0
            std = df[has_enough.index[has_enough]].pct_change().std()
            #std = df.pct_change().std()
            pos *= (1.0 / std)

        # Normalize
        pos /= pos.sum()
        
        # position scaling by kelly fraction
        kelly_output = self._get_kelly_fraction(date, wealth, model_rtn)
        weight = pos.mul(kelly_output['fr'], fill_value=0)
        
        if self.fill_cash and self._is_tradable(date, self.cash_equiv):
            pos.loc[self.cash_equiv] = 0.0
            pos.loc[self.cash_equiv] = 1.0 - pos.sum()
                
            weight.loc[self.cash_equiv] = 0.0
            weight.loc[self.cash_equiv] = 1.0 - weight.sum()
        
        return weight, pos, ranks, kelly_output
    
    
    def _get_selection(self, sig, date):
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
        #if self.cash_equiv in pos.index:
            pos.loc[self.cash_equiv] += pos_cash
            
        except:
        #else: 
            pos.loc[self.cash_equiv] = pos_cash
        
        #pos.loc[self.cash_equiv] += pos_cash
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
    
    
    def _get_kelly_fraction(self, date, wealth, model_rtn):
        out = {'fr': 1.0}

        if (self.apply_kelly is not None) and (date!=self.dates[0]):
          
            if self.apply_kelly['self_eval']:
                ref_rtn = np.array(wealth)[-1:-1-self.apply_kelly['vol_period']-1:-1,-1][::-1]
                ref_rtn = ref_rtn[1:] / ref_rtn[:-1] - 1.0
                
            else:
                ref_rtn = np.array(model_rtn)[-1:-1-self.apply_kelly['vol_period']:-1][::-1]
            
            if len(ref_rtn)>=20:
                if self.apply_kelly['method']=='semivariance':
                    up = ev._std_dir_by_r(ref_rtn, 1)/100
                    down = ev._std_dir_by_r(ref_rtn, -1)/100
                    fr_raw = ((up-down) / (2*up*down))

                    if not np.isnan(fr_raw): 
                        out['up'] = up
                        out['down'] = down
                        out['fr_raw'] = fr_raw
                        out['fr'] = fr_raw.clip(0,1)

                elif self.apply_kelly['method']=='traditional':
                    #cash_rtn = self.r[self.cash_equiv].loc[:date][-1:-1-self.apply_kelly['vol_period']:-1][::-1]
                    #mu_cash = np.nanmean(cash_rtn)
                    mu = np.nanmean(ref_rtn)
                    var = np.nanvar(ref_rtn)

                    if (not np.isnan(var)) and var!=0:
                        fr_raw = mu / var
                        out['mu'] = mu
                        out['var'] = var
                        out['fr_raw'] = fr_raw
                        out['fr'] = fr_raw.clip(0,1)

        return out