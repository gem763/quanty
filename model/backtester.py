import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn import linear_model
from sklearn.metrics import r2_score
from pandas.tseries.offsets import Day
from IPython.core.debugger import set_trace
from matplotlib.ticker import ScalarFormatter
import matplotlib.dates as mdates

sns.set_style('ticks')
#mpl.rc('font', family='NanumGothic')
mpl.rc('axes', unicode_minus=False)



class Backtester(object):
  
    def __init__(self, params, **opt):
        self.__dict__.update(self._overwrite_params(params, **opt))
        self.dates, self.dates_asof = self._get_dates()
        
        assets = set(self.assets_member.values.flatten())
        assets.update({self.beta_to, self.cash_equiv})
        
        assets_bet = set(self.assets_member.bet)
        assets_bet.update({self.cash_equiv})
        
        self.p = self.data[self.perf_src].unstack().loc[:self.end].reindex(columns=assets).fillna(method='ffill')
        self.p_ref = self.data[self.ref_src].unstack().loc[:self.end].reindex(columns=self.assets_member.ref).fillna(method='ffill')
        self.r = self.p.pct_change()
        
        p_bet = self.data[self.bet_src].unstack().loc[:self.end].reindex(columns=assets_bet).fillna(method='ffill')
        p_high = self.data['high'].unstack().loc[:self.end].reindex(columns=assets_bet).fillna(method='ffill')
        p_low = self.data['low'].unstack().loc[:self.end].reindex(columns=assets_bet).fillna(method='ffill')
        
        p_bet_high = p_bet * (p_high/p_bet).mean()
        p_bet_low = p_bet * (p_low/p_bet).mean()

        p_bet_high.update(p_high)
        p_bet_low.update(p_low)
        
        if self.trading_tolerance=='at_close':
            self.p_close = p_bet
            self.p_buy = p_bet
            self.p_sell = p_bet
            
        elif self.trading_tolerance=='buyHigh_sellLow':
            self.p_close = p_bet            
            self.p_buy = p_bet_high
            self.p_sell = p_bet_low
            
        elif self.trading_tolerance=='buyLow_sellHigh':
            self.p_close = p_bet
            self.p_buy = p_bet_low
            self.p_sell = p_bet_high
        
        #set_trace()        
        self._run()

        
    def _overwrite_params(self, base_params, **what):
        out = base_params.copy()
        out.update(what)
        return out
        
      
    # 모든 영업일 출력
    def _get_dates(self):
        dates_all = self.data.index.levels[0]
        dates = dates_all[(self.start<=dates_all) & (dates_all<=self.end)]
        
        # 무조건 첫날(start)과 마지막날(end)은 포함
        if self.start not in dates: 
            dates = dates.insert(0, pd.Timestamp(self.start))
            
        if self.end not in dates: 
            #dates = dates.insert(-1, pd.Timestamp(self.end))
            dates = dates.append(pd.DatetimeIndex([self.end]))

        dates_asof = pd.date_range(self.start, self.end, freq='M')
        dates_asof = dates_all[dates_all.get_indexer(dates_asof, method='ffill')] & dates
        
        # 무조건 첫날(start)은 리밸 기준일
        if self.start not in dates_asof: 
            dates_asof = dates_asof.insert(0, pd.Timestamp(self.start))

        return dates, dates_asof
      

    def _get_signal2(self, date):
        n_sig_w = len(self.sig_w)
        n_back = n_sig_w*31 + 40
        date_from = date - n_back*Day()
        date_to = date - 0*Day()
        
        p = self.p_ref.loc[date_from:date_to].resample('M').ffill().iloc[-n_sig_w-1:]
        r = (p.iloc[-1]/p.iloc[:-1]-1).replace(np.inf, np.nan)
        sig_w = self.sig_w[-len(r):] #[::-1][:len(r)][::-1]
        
        sig = r.mul(sig_w, axis=0).sum(skipna=False)
        
        #if date==pd.Timestamp('2016-01-29'): set_trace()
        if self.self_trend is not None:
            p_period = self.p_ref.loc[:date]
            p_ma_short = p_period.iloc[-self.self_trend[0]:].mean()
            p_ma_long = p_period.iloc[-self.self_trend[1]:].mean()
            sig[p_ma_short<p_ma_long] = np.nan
        
        sig.index = self.assets_member.bet
        return sig

      
    def _get_signal(self, date):
        n_sig_w = len(self.sig_w)
        n_back = n_sig_w*31 + 40
        date_from = date - n_back*Day()
        date_to = date - 0*Day()
        
        p = self.p_ref.loc[date_from:date_to].resample('M').ffill().iloc[-n_sig_w-1:]
        r = (p.iloc[-1]/p.iloc[:-1]-1).replace(np.inf, np.nan)
        sig_w = self.sig_w[-len(r):] #[::-1][:len(r)][::-1]
        
        sig = r.mul(sig_w, axis=0).sum(skipna=False)        
        sig.index = self.assets_member.bet
        
        not_tradable = ~self._is_tradable(date)
        sig.loc[not_tradable] = np.nan
        return sig
      
      
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
        
#         pos_cash = 0
#         if self._is_tradable(date, self.cash_equiv):
#            pos_cash = self.n_picks - pos.sum()

        pos.loc[self.cash_equiv] += pos_cash
        return pos, ranks  
        
      
    def _get_selection2(self, sig, date):
        has_ma_mom_rf = self._has_ma_mtum_single(date, self.rf_trend, self.riskfree)

        pos, ranks = self._get_default_selection(date, sig, self.n_picks)
        pos.loc[self.riskfree] = 0
        #pos.loc[self.cash_equiv] = 0 # 요건 당연히 0 일듯
        
          
        if self._is_tradable(date, self.riskfree):
            pos_rf = self.n_picks - pos.sum()

            if self.positive_sig_for_riskfree:
                if (not has_ma_mom_rf) and sig.loc[self.riskfree]<0:
                    pos_rf = 0
    
                elif (not has_ma_mom_rf) and sig.loc[self.riskfree]>=0:
                    pos_rf = int(pos_rf*0.5)
                    #pass

                elif has_ma_mom_rf and sig.loc[self.riskfree]<0:
                    pos_rf = int(pos_rf*0.5)
                    #pass
                
                elif has_ma_mom_rf and sig.loc[self.riskfree]>=0:
                    pass
                
                else:
                    pos_rf = 0

            pos.loc[self.riskfree] = pos_rf
        
        return pos, ranks
      

    def _get_kelly_fraction(self, date):
        out = {'fr': 1.0}

        if (self.apply_kelly is not None) and (date!=self.dates[0]):
          
            if self.apply_kelly['self_eval']:
                ref_rtn = np.array(self.wealth)[-1:-1-self.apply_kelly['vol_period']-1:-1,-1][::-1]
                ref_rtn = ref_rtn[1:] / ref_rtn[:-1] - 1.0
                
            else:
                ref_rtn = np.array(self.model_rtn)[-1:-1-self.apply_kelly['vol_period']:-1][::-1]
            
            if len(ref_rtn)>=20:
                if self.apply_kelly['method']=='semivariance':
                    up = self._std_dir_by_r(ref_rtn, 1)/100
                    down = self._std_dir_by_r(ref_rtn, -1)/100
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
            
    
    def _is_tradable(self, date, asset=None):
        if type(asset) is str:
            return not np.isnan(self.p_close.loc[:date, asset].iloc[-1])
          
        else:
            return self.p_close.loc[:date].iloc[-1].notnull()
    
    
    def _get_weights(self, sig, date):
        #if date==pd.Timestamp('2016-07-29'): set_trace()
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
        kelly_output = self._get_kelly_fraction(date)
        weight = pos.mul(kelly_output['fr'], fill_value=0)
        
        if self.fill_cash and self._is_tradable(date, self.cash_equiv):
            pos.loc[self.cash_equiv] = 0.0
            pos.loc[self.cash_equiv] = 1.0 - pos.sum()
                
            weight.loc[self.cash_equiv] = 0.0
            weight.loc[self.cash_equiv] = 1.0 - weight.sum()
        
        return weight, pos, ranks, kelly_output
      
      
    # 종목별 CAGR
    def _cagr(self, p):
        return ((p[-1]/p[0])**(250/(len(p)-1)) - 1) * 100


    # 종목별 변동성
    def _std(self, p): 
        return np.nanstd(p[1:]/p[:-1]-1) * (250**0.5) * 100


    # 종목별 하방변동성
    def _std_down(self, p):
        r = p[1:]/p[:-1]-1
        r = r[~np.isnan(r)]
        r_neg = r[r<0]
        return ((((r_neg**2).sum() / (len(r)-1))**0.5) * (250**0.5)) * 100

      
    def _std_dir_by_r(self, r, dir): 
        #r = p[1:]/p[:-1]-1
        #r = r[~np.isnan(r)]
        if dir==1:
            r_dir = r[r>0]
        elif dir==-1:    
            r_dir = r[r<0]
        return ((((r_dir**2).sum() / (len(r)-1))**0.5) * (250**0.5)) * 100

      
    def _std_dir(self, p, dir): 
        r = p[1:]/p[:-1]-1
        r = r[~np.isnan(r)]
        if dir==1:
            r_dir = r[r>0]
        elif dir==-1:    
            r_dir = r[r<0]
        return ((((r_dir**2).sum() / (len(r)-1))**0.5) * (250**0.5)) * 100      
      

    # 종목별 Sharpe
    def _sharpe(self, p): 
        return self._cagr(p)/self._std(p)

    # 종목별 MDD
    def _mdd(self, p):
        return (p/np.maximum.accumulate(p)-1).min() * 100
        
  
    # 종목별 수익안정성
    def _consistency(self, p):
        if len(p)==0:
            return np.nan
          
        y = np.log(p)
        X = np.arange(len(p)).reshape(-1,1)
        model = linear_model.LinearRegression(fit_intercept=False).fit(X, y)
        return r2_score(y, model.predict(X))
  
  
    # 종목별 베타
    def _beta(self, rtns):
        if len(rtns)==0:
            return np.nan
          
        y = rtns[:,0].reshape(-1,1)
        X = rtns[:,1].reshape(-1,1)
        model = linear_model.LinearRegression().fit(X, y)
        return model.coef_[0,0]
  
  
    def get_stats(self):
        cum = self.cum
        cum_last = cum.iloc[-1]
        n_samples = cum.count()
        
        # Base stats: 전체구간
        cagr = (cum_last**(250/(n_samples-1)) - 1) * 100
        std = cum.pct_change().std() * (250**0.5) * 100
        sharpe = cagr / std
        #mdd = (cum.expanding().apply(lambda x: x[-1]/np.nanmax(x)-1).min()) * 100
        mdd = (cum.div(cum.cummax()) - 1).min() * 100
        #set_trace()
        consistency = (pd.Series([self._consistency(cum[col].dropna()) for col in cum], index=cum.columns)) * 100
        beta = pd.Series([self._beta(cum[[col, self.beta_to]].pct_change().dropna(how='any').values) for col in cum], index=cum.columns)

        # Rolling stats
        cum_roll = cum.rolling(self.n_roll_stats)
        cagr_roll = cum_roll.apply(self._cagr)
        cagr_roll_med = cagr_roll.median()
        loss_proba = (cagr_roll<0).sum()/cagr_roll.count() * 100
        std_roll_med = cum_roll.apply(self._std).median()
        sharpe_roll_med = cum_roll.apply(self._sharpe).median()
        #mdd_roll_med = cum_roll.apply(self._mdd).median()

        # With 1M returns
        r_month = cum.resample('M').ffill().pct_change()
        hit = ((r_month>0).sum() / (r_month.count()-1)) * 100
        profit_to_loss = - r_month[r_month>0].mean() / r_month[r_month<0].mean()
        
        return pd.DataFrame({
            'cum_last': cum_last,
            'n_samples': n_samples, 
            'cagr': cagr, 
            'std': std,
            'sharpe': sharpe,
            'mdd': mdd,
            'cagr_roll_med': cagr_roll_med, 
            'std_roll_med': std_roll_med, 
            'sharpe_roll_med': sharpe_roll_med, 
            'beta': beta, 
            'loss_proba': loss_proba, 
            'hit': hit,
            'profit_to_loss': profit_to_loss,
            'consistency': consistency,
        }).round(2)


    def _run(self):
        trade_due = -1
        
        # 매일 기록
        self.hold = []
        self.eq_value = []
        self.wealth = []
        self.pos_d = []
        self.model_rtn = []
        self.model_contr = []
        
        # 의사결정일에만 기록
        self.sig = []
        self.ranks = []
        self.pos = []     # 듀얼모멘텀 모델 자체에서 산출되는 비중
        self.weight = []  # 최종비중 (켈리반영)
        self.kelly = []
        
        for date in self.dates:
            if date in self.p.index: 
                trade_due -= 1
                
            trade_amount_ = trade_cashflow_ = cost_ = 0
            
            hold_ = self.hold[-1].copy() if len(self.hold)>0 else pd.Series()
            cash_ = self.wealth[-1][-2] if len(self.wealth)>0 else self.cash
            weight_ = self.weight[-1].copy() if len(self.weight)>0 else pd.Series()
            pos_ = self.pos[-1].copy() if len(self.pos)>0 else pd.Series()
            pos_d_ = self.pos_d[-1].copy() if len(self.pos_d)>0 else pd.Series()
            
            
            # 0. 리밸런싱 실행하는 날
            if trade_due==0:
                trade_amount_, trade_cashflow_, cost_, cash_, hold_ = self._rebalance(date, hold_, cash_, weight_)
                pos_d_, model_rtn_, model_contr_ = self._update_pos_daily(date, pos_d_)
                pos_d_ = pos_.copy()
                

            # 1. 리밸런싱 비중결정하는 날
            elif date in self.dates_asof:
                weight_, pos_, sig_, ranks_, trade_due, kelly_output = self._positionize(date, weight_, trade_due)
                pos_d_, model_rtn_, model_contr_ = self._update_pos_daily(date, pos_d_)
                
                self.sig.append(sig_)
                self.ranks.append(ranks_)
                self.weight.append(weight_)
                self.pos.append(pos_)
                self.kelly.append(kelly_output)
                
              
            # 2. 아무일도 없는 날
            else:
                pos_d_, model_rtn_, model_contr_ = self._update_pos_daily(date, pos_d_)
                

            # 종가기준 포지션 밸류 측정
            eq_value_, value_, nav_ = self._evaluate(date, hold_, cash_)
            
            
            self.hold.append(hold_)
            self.eq_value.append(eq_value_)
            self.wealth.append([trade_amount_, value_, trade_cashflow_, cost_, cash_, nav_])
            self.pos_d.append(pos_d_)
            self.model_rtn.append(model_rtn_)
            self.model_contr.append(model_contr_)

            
        # 종목별 시그널, 포지션
        self.sig = pd.DataFrame(self.sig, index=self.dates_asof)
        self.ranks = pd.DataFrame(self.ranks, index=self.dates_asof)
        self.weight = pd.DataFrame(self.weight, index=self.dates_asof)
        self.pos = pd.DataFrame(self.pos, index=self.dates_asof)
        self.kelly = pd.DataFrame(self.kelly, index=self.dates_asof)
        
        # Daily Booking
        self.hold = pd.DataFrame(self.hold, index=self.dates)
        self.eq_value = pd.DataFrame(self.eq_value, index=self.dates)
        self.wealth = pd.DataFrame(self.wealth, index=self.dates, columns=['trade_amount', 'value', 'trade_cashflow', 'cost', 'cash', 'nav'])
        self.pos_d = pd.DataFrame(self.pos_d, index=self.dates)
        self.model_rtn = pd.Series(self.model_rtn, index=self.dates)
        self.model_contr = pd.DataFrame(self.model_contr, index=self.dates)
        
        # 지수가격(normalized)
        cum = self.p.reindex(self.dates, method='ffill')
        cum['DualMomentum'] = self.wealth['nav']
        self.cum = cum / cum.bfill().iloc[0]

        # Turnover
        self.turnover = self._get_turnover()
        
        # 성과통계
        self.stats = self.get_stats()
        
        
    def _get_turnover(self):
        turnover = pd.Series()

        for idt, dt in enumerate(self.weight.index):
            if idt!=0:
                turnover[dt] = self.weight.iloc[idt].sub(self.weight.iloc[idt-1], fill_value=0).abs().sum()/2  

        return turnover.rolling(12).sum().dropna()
    
        
    def _update_pos_daily(self, date, pos_daily_last_):
        #if date>pd.Timestamp('2005-12-31'): set_trace()
        if date in self.r.index:
            pos_cash = 1.0 - pos_daily_last_.sum()
            pos_updated = (1+self.r.loc[date,pos_daily_last_.index]).mul(pos_daily_last_, fill_value=0)
            pos_total = pos_cash + pos_updated.sum()
            pos_total = 1.0 if pos_total==0 else pos_total
            model_rtn_ = pos_total - 1.0
            has_pos = pos_daily_last_.index[pos_daily_last_!=0]
            model_contr_ = pos_updated.sub(pos_daily_last_, fill_value=0)[has_pos]
        
            return pos_updated/pos_total, model_rtn_, model_contr_
        
        else:
            return pos_daily_last_, 0.0, pd.Series()
      
                
    def _rebalance(self, date, hold_, cash_, pos_tobe_):
        if self.trade_prev_nav_based:
            pos_prev_amount = hold_*self.p_close.loc[:date-Day()].iloc[-1]
        else:
            pos_prev_amount = hold_*self.p_close.loc[date]
            
        # Planning
        nav_prev = pos_prev_amount.sum() + cash_
        pos_amount = self.gr_exposure * nav_prev * pos_tobe_
        pos_buffer = nav_prev - pos_amount.sum()
        amount_chg = pos_amount.sub(pos_prev_amount, fill_value=0)
        amount_buy_plan = amount_chg[amount_chg>0]
        amount_sell_plan = -amount_chg[amount_chg<0]

        # Sell first
        p_sell_ = self.p_sell.loc[date]
        share_sell = amount_sell_plan.div(p_sell_).dropna()
        share_sell.where(share_sell.lt(hold_), hold_, inplace=True)
        share_sell.where(pos_tobe_>0, hold_, inplace=True) # 비중 0는 완전히 팔아라
        amount_sell = share_sell*p_sell_
        amount_sell_sum = amount_sell.sum()
        cost_sell = amount_sell_sum * self.expense
        cash_ += (amount_sell_sum - cost_sell)

        # Buy next
        p_buy_ = self.p_buy.loc[date]
        amount_buy_plan_sum = amount_buy_plan.sum()
        amount_buy = amount_buy_plan * np.min([amount_buy_plan_sum, cash_-pos_buffer]) / amount_buy_plan_sum
        amount_buy_sum = amount_buy.sum()
        share_buy = amount_buy.div(p_buy_).dropna()
        cost_buy = amount_buy_sum * self.expense
        cash_ += (-amount_buy_sum - cost_buy)

        # 매매결과
        cost_ = cost_buy + cost_sell
        trade_cashflow_ = amount_sell_sum - amount_buy_sum
        trade_amount_ = amount_sell_sum + amount_buy_sum

        # 최종포지션
        hold_ = hold_.add(share_buy, fill_value=0).sub(share_sell, fill_value=0).dropna()
        
        return trade_amount_, trade_cashflow_, cost_, cash_, hold_
                
        
    def _positionize(self, date, weight_asis_, trade_due):
        sig_ = self._get_signal(date)
        weight_, pos_, ranks_, kelly_output = self._get_weights(sig_, date)
        
        if weight_.sub(weight_asis_, fill_value=0).abs().sum()!=0:
            trade_due = self.trade_delay
            
        return weight_, pos_, sig_, ranks_, trade_due, kelly_output
                    
    
    def _evaluate(self, date, hold_, cash_):
        if date==self.dates[0]:
            eq_value_ = pd.Series()
        else:
            eq_value_ = hold_ * self.p_close.loc[:date].iloc[-1]

        value_ = eq_value_.sum()
        nav_ = value_ + cash_
        return eq_value_, value_, nav_
        
        
    def plot_cum(self, strats, **params):
        plot_cum(self.cum, strats, **params)
        
        
    def plot_cum_yearly(self, strats, **params): 
        plot_cum_yearly(self.cum[strats], **params)

        
    def plot_turnover(self):
        #plot_turnover(self.weight)
        plot_turnover(self.turnover)
        
        
    def plot_weight(self, rng): 
        plot_weight(self.weight, rng, self.riskfree, self.cash_equiv)
        
        
    def plot_stats(self, strats, style=None, **params):
        if style is None:
            items = {
                'cagr': 'CAGR(%)', 
                'std': 'Standard dev(%)', #'연변동성(%)', 
                'sharpe': 'Sharpe', 
                'cagr_roll_med': 'CAGR(%,Rolling1Y)', 
                'std_roll_med': 'Standard dev(%,Rolling1Y)', 
                'sharpe_roll_med': 'Sharpe(Rolling1Y)', 
                'mdd': 'MDD(%)', 
                'hit': 'Hit ratio(%,1M)', 
                'profit_to_loss': 'Profit-to-loss(%,1M)', #'평균손익비(%,1M)', 
                'beta': 'Beta(vs.' + self.beta_to + ')',
                'loss_proba': 'Loss probability(%,1Y)', #'손실확률(%,1Y)', 
                'consistency': 'Consistency(%)',
            }
            
            plot_stats(self.stats, strats, items, **params)
        
        elif style=='simple':
            items = {
                'cagr': 'CAGR(%)', 
                'std': 'Standard dev(%)', #'연변동성(%)', 
                'sharpe': 'Sharpe', 
                'mdd': 'MDD(%)', 
            }          
        
            plot_stats(self.stats, strats, items, ncols=4, **params)
        
        
    def plot_profile(self, strats, **params):
        plot_profile(self.stats, strats, **params)
        
        
    def plot_dist(self, strats, **params):
        items = {
            self._cagr: 'CAGR(%,Rolling1Y)', 
            self._std: 'Standard dev(%,Rolling1Y)',
            self._sharpe: 'Sharpe(Rolling1Y)', 
        }
        
        plot_dist(self.cum, strats, items, **params)
        
        
    def plot_contr_cum(self, **params):
        plot_contr_cum(self.model_contr, **params)
        
        
    def plot_breakdown(self):
        plot_breakdown(self.model_contr, self.weight)
        
        
        
        
def plot_contr_cum(contr, assets=None):
    if assets is None:
        contr_cum = contr.add(1, fill_value=0).cumprod()
    else: 
        contr_cum = contr[assets].add(1, fill_value=0).cumprod()

    contr_cum.plot(figsize=(20,10))
        

def plot_cum(prices, strats, names=None, color=None, style=None, logy=True):
    #plt.figure()
    
    prices_ = prices[strats]
    ax = prices_.plot(
        figsize=(7,5), 
        logy=logy, color=color, style=style, 
        xlim=(prices_.index[0], prices_.index[-1]), 
    )

    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel('')
    
    legend_fsize = 12
    if names: ax.legend(names, fontsize=legend_fsize)
    else: ax.legend(fontsize=legend_fsize)
        

def plot_cum_yearly(cum, names=None, color=None, style=None, remove=[]):
    years = cum.index.year.unique()
    eoy = None
    cum_list = []

    for iyear, year in enumerate(years):
        cum_ = cum.loc[eoy:str(year)]

        if (len(cum_)>1) and (year not in remove):
            cum_ /= cum_.iloc[0]
            cum_list.append((year, cum_))

        eoy = cum_.index[-1]


    nFig = len(cum_list)
    nWidth = 5
    nHeight = int(np.ceil(float(nFig)/nWidth))
    fSize = 2.5

    fig, axes = plt.subplots(nHeight, nWidth, sharey=True, figsize=(fSize*nWidth, fSize*nHeight))
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.6)
    [ax.axis('off') for ax in axes]

    for i, (year, cum_) in enumerate(cum_list):
        ax = axes[i]
        ax.axis('on')
        cum_.plot(ax=ax, legend=False, xticks=cum_.index[::60], color=color, style=style, xlim=(cum_.index[0],cum_.index[-1]))
        ax.set_title(year, fontsize=15, weight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))

    if names is None:
        names = strats
        
    axes[0].legend(names, bbox_to_anchor=(0, 1.2, nWidth, 0), ncol=len(strats), loc=3);
    
    
def plot_turnover(turnover):    
#def plot_turnover(weights):
#     turnover = pd.Series()

#     for idt, dt in enumerate(weights.index):
#         if idt!=0:
#             turnover[dt] = weights.iloc[idt].sub(weights.iloc[idt-1], fill_value=0).abs().sum()/2  
      
#     turnover_12m = turnover.rolling(12).sum().dropna()

#     turnover_avg = turnover.mean()
    ax = turnover.plot(ylim=(0,10), color='k', xlim=(turnover.index[0], turnover.index[-1]))
    ax.set_title('Turnover ratio (12M)', fontsize=15, weight='bold')
    ax.axhline(turnover.mean(), color='k', linestyle='--', linewidth=1);
      

def plot_breakdown(model_contr, weight):
    contr = model_contr.mean()
    contr /= contr.sum()
    p_break = pd.DataFrame()
    p_break['contr'] = contr*100
    p_break['n_month'] = (weight!=0).sum()
    p_break = p_break.sort_values(by=['contr'])

    ax = p_break.plot.barh(
            subplots=True, legend=False, sharex=False, sharey=True, width=0.8,
            figsize=(7, len(p_break)/3.0), 
            layout=(1, 2), 
            #color=('k', 'k'), 
            edgecolor='k', 
            lw=1, 
    )

    ax[0,0].set_title('Contribution (Total=100)', fontsize=15, weight='bold')
    ax[0,1].set_title('# of months', fontsize=15, weight='bold')
    ax[0,0].axvline(0, color='k', linestyle='-', linewidth=1)  
      

def plot_weight(weight, rng, riskfree, cash_equiv):
    weight_ = weight.copy().drop([cash_equiv], axis=1)
    weight_i = weight_.index + 5*Day()
    weight_.index = weight_i
    weight_ = weight_[str(rng[0]):str(rng[1])]
    weight_.index = weight_.index.strftime('%Y-%m')
    
    weight__ = []
    for dt in weight_.index:
        has_weight = weight_.loc[dt].abs() > 0.001
        weight__.append(weight_.loc[dt][has_weight])
    
    weight__ = pd.DataFrame(weight__)
    cols = list(weight__.columns)
    cols.remove(riskfree)
    cols = [riskfree] + cols
    weight__ = weight__[cols]

    bar_w = 0.8
    fig_h = len(weight__)/3.0
    ax = weight__.plot.barh(stacked=True, figsize=(10,fig_h), colormap='tab20c', width=bar_w, xlim=(0,1))
    ax.legend(loc=1, bbox_to_anchor=(1.25, 1));
    
      

def plot_stats(stats, strats, items, names=None, color=None, lim=None, ncols=3):
    #plt.figure()
        
    height_strats = 0.6 # 전략별 bar 높이
    n_items = len(items)
    n_cols = ncols
    n_rows = int(np.ceil(n_items/float(n_cols)))
    fig_width = ncols * 8.0/3.0 #8
      
    stats_ = stats.loc[strats, items.keys()]#.copy()
    if names: stats_.index = names
    if color: color = [color] * n_items
      
    ax = stats_.plot.barh(
        subplots=True, legend=False, sharex=False, sharey=True, width=0.8,
        figsize=(fig_width, height_strats*len(strats)*n_rows), 
        layout=(n_rows, n_cols), 
        title=items.values(), 
        color=color, 
        edgecolor='k', 
        lw=1,
        #xerr=err_value, 
    )
    
    for i, ax_ in enumerate(ax.flatten()):
        if lim: ax_.set_xlim(lim[i])
        ax_.axvline(0, color='k', linestyle='-', linewidth=1)
    
    plt.subplots_adjust(hspace=1.5)#0.7)
    
    
def plot_profile(stats, strats, names=None, color=None, bsize=None):
    #plt.figure()
    
    cagr = stats['cagr']
    std = stats['std']

    # 차트범위 최대값
    lim = np.ceil(max(cagr.max(), std.max()) * 1.1 / 5) * 5

    # 듀얼모멘텀을 지나는 직선들
    x0, y0 = std[strats], cagr[strats]
    slope = y0 / x0
    X_ = np.linspace(0, lim, 100)
    Y_ = slope.values * X_.reshape(-1,1)
    ax = pd.DataFrame(Y_, index=X_).plot(zorder=-1, style='k-', legend=False)
        
    # 위험조정수익률=1 인 직선
    pd.Series(X_, index=X_).plot(zorder=-1, style='k--', legend=False, ax=ax)

    # color 설정
    i_strats = stats.index.get_indexer(strats)
    c_ = np.full(len(std), None)
    c_[:] = 'k'
    if color: c_[i_strats] = color

    # 버블 사이즈 설정
    s_ = np.full(len(std), None)
    s_[:] = 100
    if bsize: s_[i_strats] = bsize
    
    # 라벨 설정
    labels = stats.index.values.copy() # copy안하면 원래 index가 바뀌어버린다
    if names: labels[i_strats] = names

    # Scatter plot
    stats.plot.scatter(
        x='std', y='cagr', ax=ax, edgecolor='k', 
        xlim=(0,lim), ylim=(0,lim), figsize=(7,7), 
        s=s_.tolist(), 
        c=c_.tolist(), 
        lw=1,
    )

    # Annotation
    for label, x, y in zip(labels, std, cagr):
        ax.annotate(
            label, 
            xy=(x,y), 
            xytext=(5,5),
            textcoords='offset points', 
            ha='left', #'right', 
            va='bottom',
            bbox=dict(facecolor='w', alpha=0.8, lw=1), 
            size=12,
        )

    ax.set_xlabel('Standard deviation(%)', size=15) # 연변동성
    ax.set_ylabel('CAGR%)', size=15)
    
    
def plot_dist(prices, strats, items, n_roll_stats=250, names=None, color=None):
    #plt.figure()
    
    height_strats = 1.5
    prices_ = prices[strats]#.copy()
    if names: prices_.columns = names

    fig, axes = plt.subplots(len(strats), len(items), figsize=(11,height_strats*len(strats)))
    prices_rolled = prices_.rolling(n_roll_stats)

    for i, (item_, label_) in enumerate(items.items()):
        collected = prices_rolled.apply(item_)
        med = collected.median()
        legend = True if i==0 else False

        ax = collected.plot.hist(
            bins=50, edgecolor='k', subplots=True, 
            sharex=True, histtype='stepfilled', 
            color=color, 
            ax=axes[:,i], 
            legend=legend, 
            lw=1,
        )

        for j, ax_ in enumerate(ax):
            ax_.axvline(0, color='k', linestyle='--', linewidth=1)
            ax_.axvline(med[j], color='r', linewidth=5, alpha=0.5)
            ax_.set_ylabel('')

        ax[-1].set_xlabel(label_, size=15)
        
        
def plot_stats_pool(stats_pool, items, names=None, lim=None):
    stats_pool_ = stats_pool.loc[:,items.keys()]#.sort_index()
    if names: stats_pool_.index = names
    
    f_height = 1.5
    ax = stats_pool_.plot.bar(
        subplots=True, sharex=True, sharey=False, legend=False, 
        width=0.8, color='k', 
        layout=(len(items),1), 
        figsize=(5,f_height*len(items)), 
        title=items.values(), 
    )
    
    for i, ax_ in enumerate(ax): 
        if lim: ax[i,0].set_ylim(lim[i])
        #ax[i,0].set_title(fontsize=15, weight='bold')

    plt.subplots_adjust(hspace=0.5)        