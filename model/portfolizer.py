import numpy as np
import pandas as pd
from ..model import evaluator as ev


class Portfolio(object):
    def __init__(self, w_type, cash_equiv, p_close, iv_period, apply_kelly):
        self.w_type = w_type
        self.cash_equiv = cash_equiv
        self.p_close = p_close
        self.iv_period = iv_period
        self.apply_kelly = apply_kelly
        
        
    def get(self, selection, date, sig, ranks, wealth, model_rtn):
        
        if self.w_type=='ew':
            pos = selection

        elif self.w_type=='ranky':
            pos = selection / ranks

        elif self.w_type=='ranky2':
            pos = selection / (ranks**0.5)
            
        elif self.w_type=='sig':
            sig_ = sig[selection!=0]
            sig_[sig_<=0] = sig_[sig_>0].mean()
            pos = selection * sig_
            
        elif self.w_type=='iv':
            i_date = self.p_close.index.get_loc(date, method='ffill')
            df = self.p_close.iloc[i_date+1-self.iv_period:i_date+1]
            
            # Underlying index 중에 종종 일일데이터가 없는 경우가 있다
            # 이 종목은 변동성이 매우 작을 것이므로, 제거한다
            has_enough = pd.Series({k:df[k].nunique() for k in df}) > self.iv_period/2.0
            std = df[has_enough.index[has_enough]].pct_change().std()
            #std = df.pct_change().std()
            pos = selection / std

        # Normalize
        pos /= pos.sum()
        
        # position scaling by kelly fraction
        kelly_output = self._get_kelly_fraction(date, wealth, model_rtn)
        weight = pos.mul(kelly_output['fr'], fill_value=0)
        
        #if self._is_tradable(date, self.cash_equiv):# and self.fill_cash:
        pos.loc[self.cash_equiv] = 0.0
        pos.loc[self.cash_equiv] = 1.0 - pos.sum()
                
        weight.loc[self.cash_equiv] = 0.0
        weight.loc[self.cash_equiv] = 1.0 - weight.sum()
        
        return weight, pos, kelly_output

    
    
    def _get_kelly_fraction(self, date, wealth, model_rtn):
        out = {'fr': 1.0}

        if (self.apply_kelly is not None) and (len(wealth)>0) and (len(model_rtn)>0): #(date!=self.dates[0]):
          
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