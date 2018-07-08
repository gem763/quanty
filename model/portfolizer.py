import numpy as np
import pandas as pd
from ..model import evaluator as ev
from IPython.core.debugger import set_trace


class Portfolio(object):
    #def __init__(self, w_type, cash_equiv, p_close, iv_period, apply_kelly, r, bm, safety_buffer, te_target):
    def __init__(self, **params):
        self.__dict__.update(**params)
        #self.w_type = w_type
        #self.cash_equiv = cash_equiv
        #self.p_close = p_close
        #self.iv_period = iv_period
        #self.apply_kelly = apply_kelly
        #self.r = r
        #self.bm = bm
        #self.safety_buffer = safety_buffer
        #self.te_target = te_target
        self.wr = []
        self.wc = []
        
        
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
            
            
        elif self.w_type=='eaa_true':
            pr = self.p_close
            rt = pr.loc[pr.index<date, sig.index].iloc[-251:].pct_change().iloc[1:]
            #rt_short = rt.iloc[-20:]
            rt_ew = rt.mean(axis=1)
            cor = rt.corrwith(rt_ew)
            #cor = rt_short.cov().dot(np.ones(len(sig)))/rt_short.std()/((rt_short.cov().sum().sum())**0.5)
            
            #sig -= sig[self.cash_equiv]
            eaa_wr = self.eaa_wr + 1e-6 if self.eaa_wr==0 else self.eaa_wr
            score = (sig**eaa_wr) * ((1-cor)**self.eaa_wc)
            ranks = score.rank(ascending=False, na_option='bottom')
            sel = (sig>0) & (score>0) & (ranks<1+self.n_picks)
            sel = sel.astype(int)
            sel.loc[self.cash_equiv] += (self.n_picks - sel.sum())
            
            pos = sel * score

            
        elif self.w_type=='eaa_true_optima':
            pr = self.p_close
            rt = pr.loc[pr.index<date, sig.index].iloc[-251:].pct_change().iloc[1:]
            rt_ew = rt.mean(axis=1)
            cor = rt.corrwith(rt_ew)

            wrs = np.linspace(-3,3,7)
            wcs = np.linspace(-3,3,7)
            rt_short = rt.iloc[-250:]            

            #sig[~(sig>0)] = sig[sig>0].mean()            
            
            #sig -= sig[self.cash_equiv]
            std = rt_short.std()
            
            def score_(wr_, wc_):
                wr_ = wr_ + 1e-6 if wr_==0 else wr_
                score = ((sig**wr_) * ((1-cor))) / (std**wc_)
                ranks = score.rank(ascending=False, na_option='bottom')
                sel = (sig>0) & (score>0) & (ranks<1+self.n_picks)
                sel = sel.astype(int)
                sel.loc[self.cash_equiv] += (self.n_picks - sel.sum())
                pos_ = sel * score            
                pos_.fillna(0, inplace=True)
                pos_ /= pos_.sum() 
                rt_expected = (score * pos_).sum()
                #rt_expected = (sig * pos_).sum()
                #rt_expected = (rt_short.sum() * pos_).sum()
                #vol = (pos_.T.dot(rt_short.cov()).dot(pos_))**0.5
                return rt_expected# / vol#- (pos_**2).sum()
            
            candidates = np.array([[score_(wr_, wc_) for wc_ in wcs] for wr_ in wrs])
            i_wr, i_wc = np.unravel_index(candidates.argmax(), candidates.shape)
            wr = wrs[i_wr]
            wc = wcs[i_wc]
         
            wr = wr + 1e-6 if wr==0 else wr
            score = ((sig**wr) * ((1-cor))) / (std**wc)
            ranks = score.rank(ascending=False, na_option='bottom')
            sel = (sig>0) & (score>0) & (ranks<1+self.n_picks)
            sel = sel.astype(int)
            sel.loc[self.cash_equiv] += (self.n_picks - sel.sum())
            pos = sel * score            
            pos.fillna(0, inplace=True)            

            self.wr.append(wr)
            self.wc.append(wc)
            
            
            
            
        elif self.w_type=='eaa':
            pr = self.p_close
            rt = pr.loc[pr.index<date, selection.index].iloc[-251:].pct_change().iloc[1:]
            rt_ew = rt.mean(axis=1)
            cor = rt.corrwith(rt_ew)
            sig_ = sig[selection!=0]
            sig_[~(sig_>0)] = sig_[sig_>0].mean()
            pos = selection * ((sig_**self.eaa_wr) * ((1-cor)**self.eaa_wc))
            
            
        elif self.w_type=='eaa_optima': 
            pr = self.p_close
            rt = pr.loc[pr.index<date, selection.index].iloc[-251:].pct_change().iloc[1:]
            rt_ew = rt.mean(axis=1)
            cor = rt.corrwith(rt_ew)

            wrs = np.linspace(-5,5,11)
            wcs = np.array([1])
            rt_short = rt.iloc[-20:]
            
            sig_ = sig[selection!=0]
            sig_[~(sig_>0)] = sig_[sig_>0].mean()
            #sig_ -= sig.loc[self.riskfree]
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
            
            pos = selection * ((sig_**wr) * ((1-cor)**wc))
            

        
        # Normalize
        pos /= pos.sum()
        
        # position scaling by kelly fraction
        kelly_output = self._get_kelly_fraction(date, wealth, model_rtn)
        
        if self.apply_cp:
            weight = pos.mul(sum(sig>0)/float(len(sig)), fill_value=0)
        else:
            weight = pos.mul(kelly_output['fr'], fill_value=0)
        
        
        #if self._is_tradable(date, self.cash_equiv):# and self.fill_cash:
        pos.loc[self.cash_equiv] = 0.0
        pos.loc[self.cash_equiv] = 1.0 - pos.sum()

        weight.loc[self.cash_equiv] = 0.0
        weight.loc[self.cash_equiv] = 1.0 - weight.sum()

        
        if self.te_target is not None:
            te_hist = self._get_te_hist(date, wealth)
            te_exante = self._get_te_exante(date, weight, 250)
            #te_exante_short = self._get_te_exante_ewm(date, weight, 250)

            k = 0.3
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
                te_exante_short = self._get_te_exante(date, weight, 20)

                if te_exante_short==0:
                    eta_max = 1
                else:
                    eta_max = np.min([self.safety_buffer*self.te_target / te_exante_short, 1])
                
                eta = np.min([eta, eta_max])
                
                
            weight = weight.mul(eta, fill_value=0)
            
            if self.bm in weight.index:
                weight[self.bm] += (1-eta)
            else:
                weight[self.bm] = (1-eta)
        
        else:
            eta = 1
                
        return weight, pos, kelly_output, eta

    
    def _get_te_exante(self, date, weight, n_period):
        w_p = weight[weight!=0]
        if self.bm not in w_p.index: w_p[self.bm] = 0
        w_bm = pd.Series({self.bm:1.0}, index=w_p.index).fillna(0)
        cov = self.r.loc[self.r.index<date, w_p.index].iloc[-n_period:].cov() * 250
        w_diff = w_p - w_bm
        return w_diff.T.dot(cov.dot(w_diff))**0.5
    

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