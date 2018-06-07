import numpy as np
import pandas as pd
from ..model import evaluator as ev
from IPython.core.debugger import set_trace


class Portfolio(object):
    def __init__(self, w_type, cash_equiv, p_close, iv_period, apply_kelly, r, bm, safety_ratio, te_target):
        self.w_type = w_type
        self.cash_equiv = cash_equiv
        self.p_close = p_close
        self.iv_period = iv_period
        self.apply_kelly = apply_kelly
        self.r = r
        self.bm = bm
        self.safety_ratio = safety_ratio
        self.te_target = te_target
        
        
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
        
        te_hist = self._get_te_hist(date, wealth)
        te_exante = self._get_te_exante(date, weight)
        
        d = 250
        d_h = np.min([230, len(wealth)])
        d_f = 250 - d_h
        
        eta = (d*((self.safety_ratio*self.te_target)**2) - d_h*(te_hist**2)) / (d_f*(te_exante**2))
        #eta = (d * (self.safety_ratio*self.te_target)**2) / (d * (te_exante**2))
        #eta = eta**0.5
        
        if eta<0:
            eta = 0
        elif eta>1:
            eta = 1
        else:
            eta = eta**0.5
        
        #eta = eta**0.5 if eta<1.0 else 1.0
        #set_trace()
        weight = weight.mul(eta, fill_value=0)
        if self.bm in weight.index:
            weight[self.bm] += (1-eta)
        else:
            weight[self.bm] = (1-eta)
        
        return weight, pos, kelly_output, te_hist, te_exante, eta

    
    def _get_te_exante(self, date, weight):
        #set_trace()
        w_p = weight[weight!=0]
        if self.bm not in w_p.index: w_p[self.bm] = 0
        w_bm = pd.Series({self.bm:1.0}, index=w_p.index).fillna(0)
        cov = self.r.loc[self.r.index<date, w_p.index].iloc[-250:].cov() * 250
        w_diff = w_p - w_bm
        return w_diff.T.dot(cov.dot(w_diff))**0.5
    
    
    def _get_te_hist(self, date, wealth):
        if len(wealth)==0:
            return 0
        
        else:
            p_port = np.array(wealth)[-1:-1-251:-1,-1][::-1]
            r_port = p_port[1:] / p_port[:-1] - 1.0
            r_bm = self.r.loc[self.r.index<date, self.bm].iloc[-len(r_port):]
            #set_trace()
            return np.nanstd(r_port-r_bm) * (250**0.5)
    
    
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