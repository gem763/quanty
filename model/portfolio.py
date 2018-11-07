import numpy as np
import pandas as pd
from ..model import evaluator as ev
from IPython.core.debugger import set_trace
import time


# -----------------------------------------------------------------
# Port를 상속하려면, __init__과 _get_pos 를 반드시 오버라이딩 한다.
# ------------------------------------------------------------------


class Port(object):

    def __init__(self, dates_port, wealth=None, model_rtn=None, **params):
        self.__dict__.update(**params)
        weight = []
        pos = []
        eta = []
        
        for date in dates_port:
            weight_, pos_, eta_ = self._get(date, wealth, model_rtn)
            weight.append(weight_)
            pos.append(pos_)
            eta.append(eta_)
            
        self.weight = pd.DataFrame(weight, index=dates_port)
        self.pos = pd.DataFrame(pos, index=dates_port)
        self.eta = pd.DataFrame(eta, index=dates_port)
        
        #self.wr = []
        #self.wc = []
        #self.downrisk = []
        #self.uprisk = []
    
    
    def portfolize(self, date, book=None):
        return self.weight.loc[date]
    
    
    def _get_pos(self, date):
        raise NotImplementedError
        
    
    def _get(self, date, wealth, model_rtn):
        pos = self._get_pos(date)
        pos = pos.clip_upper(self.w_max)

        #set_trace()
        weight, pos = self._cash_control(date, pos, wealth, model_rtn)
        weight = self._weight_to_trade(weight, date)
        weight, eta = self._te_control(date, weight, wealth)
        
        return weight, pos, eta


    def _set_weight(self, weight, asset, w):
        try:
            weight.loc[asset] += w
        except:
            weight.loc[asset] = w

        return weight
    
    
    def _weight_to_trade(self, weight, date):
        trade_assets = dict(self.trade_assets)
        
        for asset in weight[weight!=0].index:
            asset_weight = weight[asset]
            
            if asset in trade_assets:
                if asset in trade_assets[asset].keys():
                    for k,v in trade_assets[asset].items():
                        if self._is_tradable(date, k):
                            if k==asset:
                                weight[k] = asset_weight*v
                            else: 
                                weight = self._set_weight(weight, k, asset_weight*v)

                else:
                    for k,v in trade_assets[asset].items():
                        if self._is_tradable(date, k):
                            w_ = asset_weight*v
                            weight = self._set_weight(weight, k, w_)
                            asset_weight -= w_
                            
                    if self._is_tradable(date, asset):
                        weight[asset] = asset_weight
                
        return weight
    
    
    def _is_tradable(self, date, asset):
        #return not np.isnan(self.p_close.loc[date, asset])
        return not np.isnan(self.p_close.loc[:date, asset].iloc[-1])
    
    
    def _te_control(self, date, weight, wealth):
        if self.te_target is not None:
            te_hist = self._get_te_hist(date, wealth)
            te_exante = self._get_te_exante(date, weight, 250)

            k = self.te_k
            d = 250
            d_h = np.min([230, len(wealth)])
            d_f = 250 - d_h

            if te_exante==0:
                eta = 1
            else:
                eta = (d*((self.safety_buffer*self.te_target)**2) - d_h*(te_hist**2)) / (d_f*(te_exante**2))


            if self.te_smoother and eta<k:
                eta = (k**0.5) * np.exp(eta/(2*k) - 0.5)

            elif (not self.te_smoother) and eta<0:
                eta = 0
                
            elif eta>1:
                eta = 1

            else:
                eta = eta**0.5

            
            if self.te_short_target_cap:
                te_exante_short = self._get_te_exante(date, weight, self.te_short_period)
                
                if te_exante_short==0:
                    eta_max = 1
                #elif te_exante_short<self.te_target/(2**0.5):
                #    eta_max = 1
                else:
                    eta_max = np.min([self.safety_buffer*self.te_target / te_exante_short, 1])
                
                eta = np.min([eta, eta_max])


            if self.te_short_up_down_ratio_cap:
                te_exante_short_up = self._get_te_exante_semi(date, weight, self.te_short_period, opts='up')
                te_exante_short_down = self._get_te_exante_semi(date, weight, self.te_short_period, opts='down')
                self.downrisk.append(te_exante_short_down)
                self.uprisk.append(te_exante_short_up)
                
                if te_exante_short_up==0 or te_exante_short_down==0:
                    eta_max = 1
                else:
                    #eta_max = np.min([te_exante_short_up/te_exante_short_down, 1])
                    eta_max = np.min([self.safety_buffer*te_exante_short_up/te_exante_short_down, 1])
                
                eta = np.min([eta, eta_max])
                
                
            weight = weight.mul(eta, fill_value=0)
            
            if self.bm in weight.index:
                weight[self.bm] += (1-eta)
            else:
                weight[self.bm] = (1-eta)
        
        else:
            eta = 1
    
        return weight, eta
    
    
    
    #def _cash_control(self, date, pos, sig, wealth, model_rtn):
    def _cash_control(self, date, pos, wealth, model_rtn):
       
        #if self.cm_method=='cp':
        #    weight = pos.mul(sum(sig>0)/float(len(sig)), fill_value=0)
            
        if self.cm_method=='kelly':
            kelly_output = self._get_kelly_fraction(date, wealth, model_rtn)
            weight = pos.mul(kelly_output['fr'], fill_value=0)
            
        elif self.cm_method=='up_down_ratio':
            vol_up = self._get_vol_exante_semi(date, pos, self.up_down_ratio_period, opts='up')
            vol_down = self._get_vol_exante_semi(date, pos, self.up_down_ratio_period, opts='down')
            t_weight = np.min([vol_up/vol_down, 1])
            weight = pos.mul(t_weight, fill_value=0)
            
        elif self.cm_method==None:
            weight = pos.copy()
                
        if self._is_tradable(date, self.cash_equiv):                
            pos.loc[self.cash_equiv] = 0.0
            pos.loc[self.cash_equiv] = 1.0 - pos.sum()

            weight.loc[self.cash_equiv] = 0.0
            weight.loc[self.cash_equiv] = 1.0 - weight.sum()
            
        
        weight[abs(weight)<0.0001] = 0.0
        return weight, pos
        
        
        
        
    
    def _get_te_exante(self, date, weight, n_period):
        w_p = weight[weight!=0]
        if self.bm not in w_p.index: w_p[self.bm] = 0
        w_bm = pd.Series({self.bm:1.0}, index=w_p.index).fillna(0)
        cov = self.r.loc[self.r.index<date, w_p.index].iloc[-n_period:].cov() * 250
        w_diff = w_p - w_bm
        return (w_diff.T.dot(cov.dot(w_diff)))**0.5
    


    def _get_vol_exante_semi(self, date, weight, n_period, opts):
        w_p = weight[weight!=0]
        
        if opts=='up':
            mm = self.r.loc[self.r.index<date, w_p.index].iloc[-n_period:].clip_lower(0)
        elif opts=='down':
            mm = self.r.loc[self.r.index<date, w_p.index].iloc[-n_period:].clip_upper(0)
            
        cov = mm.T.dot(mm) * 250 / n_period
        return (w_p.T.dot(cov.dot(w_p)))**0.5
    
    
    
    def _get_te_exante_semi(self, date, weight, n_period, opts):
        w_p = weight[weight!=0]
        if self.bm not in w_p.index: w_p[self.bm] = 0
        w_bm = pd.Series({self.bm:1.0}, index=w_p.index).fillna(0)
        
        
        if opts=='up':
            mm = self.r.loc[self.r.index<date, w_p.index].iloc[-n_period:].clip_lower(0)
        elif opts=='down':
            mm = self.r.loc[self.r.index<date, w_p.index].iloc[-n_period:].clip_upper(0)
            
        cov = mm.T.dot(mm) * 250 / n_period
        w_diff = w_p - w_bm
        return (w_diff.T.dot(cov.dot(w_diff)))**0.5
    

    def _get_te_exante_ewm(self, date, weight, halflife):
        #set_trace()
        w_p = weight[weight!=0]
        if self.bm not in w_p.index: w_p[self.bm] = 0
        w_bm = pd.Series({self.bm:1.0}, index=w_p.index).fillna(0)
        cov = self.r.loc[self.r.index<date, w_p.index].iloc[-halflife:].ewm(halflife=halflife).cov()
        cov = cov.loc[cov.index.levels[0][-1]]* 250
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

        if (len(wealth)>0) and (len(model_rtn)>0): #(date!=self.dates[0]):
          
            if self.kelly_self_eval:
                ref_rtn = np.array(wealth)[-1:-1-self.kelly_vol_period-1:-1,-1][::-1]
                ref_rtn = ref_rtn[1:] / ref_rtn[:-1] - 1.0
                
            else:
                ref_rtn = np.array(model_rtn)[-1:-1-self.kelly_vol_period:-1][::-1]
            
            if len(ref_rtn)>=20:
                if self.kelly_type=='semivariance':
                    up = ev._std_dir_by_r(ref_rtn, 1)/100
                    down = ev._std_dir_by_r(ref_rtn, -1)/100
                    fr_raw = ((up-down) / (2*up*down))

                    if not np.isnan(fr_raw): 
                        out['up'] = up
                        out['down'] = down
                        out['fr_raw'] = fr_raw
                        out['fr'] = fr_raw.clip(0,1)

                elif self.kelly_type=='traditional':
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