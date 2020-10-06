import numpy as np
import pandas as pd
from ..model import evaluator as ev
from IPython.core.debugger import set_trace


class Portfolio(object):

    def __init__(self, **params):
        self.__dict__.update(**params)
        self.wr = []
        self.wc = []
        self.downrisk = []
        self.uprisk = []
        
    
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
    
        
        
    def get(self, date, dm, wealth, model_rtn):
        #if date==pd.Timestamp('2003-02-28'): set_trace()
        selection = dm.selection.loc[date]
        sig = dm.sig.loc[date]
        ranks = dm.ranks.loc[date]

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
        # set_trace()
        pos /= pos.sum()
        pos = pos.fillna(0).clip_upper(self.w_max)

        weight, pos = self._cash_control(date, pos, sig, wealth, model_rtn)
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
                #if date==pd.Timestamp('2003-02-28'): set_trace()
                
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
        return not np.isnan(self.p_close.loc[date, asset])
    
    
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
    
    
    
    def _cash_control(self, date, pos, sig, wealth, model_rtn):
       
        if self.cm_method=='cp':
            weight = pos.mul(sum(sig>0)/float(len(sig)), fill_value=0)
            
        elif self.cm_method=='kelly':
            kelly_output = self._get_kelly_fraction(date, wealth, model_rtn)
            weight = pos.mul(kelly_output['fr'], fill_value=0)
            
        elif self.cm_method=='up_down_ratio':
            vol_up = self._get_vol_exante_semi(date, pos, self.up_down_ratio_period, opts='up')
            vol_down = self._get_vol_exante_semi(date, pos, self.up_down_ratio_period, opts='down')
            t_weight = np.min([vol_up/vol_down, 1])
            weight = pos.mul(t_weight, fill_value=0)
            
        elif self.cm_method==None:
            weight = pos.copy()
                
        pos.loc[self.cash_equiv] = 0.0 #[self._bet_of(self.cash_equiv)] = 0.0
        pos.loc[self.cash_equiv] = 1.0 - pos.sum() #self._bet_of(self.cash_equiv)] = 1.0 - pos.sum()

        weight.loc[self.cash_equiv] = 0.0 #self._bet_of(self.cash_equiv)] = 0.0
        weight.loc[self.cash_equiv] = 1.0 - weight.sum() #self._bet_of(self.cash_equiv)] = 1.0 - weight.sum()
        
        weight[abs(weight)<0.0001] = 0.0
        return weight, pos

    
    #def _bet_of(self, asset):
    #    if asset in dict(self.overwrite_to_bet):
    #        return dict(self.overwrite_to_bet)[asset]
    #    else:
    #        return asset
        
    
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