import pandas as pd
import numpy as np
from .plotter import Plotter as pltr
from .evaluator import Evaluator as ev
from tqdm import tqdm
from pandas.tseries.offsets import Day
from IPython.core.debugger import set_trace


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
            
    
    def _is_tradable(self, date, asset=None):
        if type(asset) is str:
            return not np.isnan(self.p_close.loc[:date, asset].iloc[-1])
          
        else:
            return self.p_close.loc[:date].iloc[-1].notnull()
    
    
    def _get_weights(self, sig, date):
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
        self.stats = ev.get_stats(self.cum, self.beta_to, self.n_roll_stats)
        
        
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
        pltr.plot_cum(self.cum, strats, **params)
        
        
    def plot_cum_yearly(self, strats, **params): 
        pltr.plot_cum_yearly(self.cum[strats], **params)

        
    def plot_turnover(self):
        pltr.plot_turnover(self.turnover)
        
        
    def plot_weight(self, rng): 
        pltr.plot_weight(self.weight, rng, self.riskfree, self.cash_equiv)
        
        
    def plot_stats(self, strats, style=None, **params):
        if style is None:
            items = {
                'cagr': 'CAGR(%)', 
                'std': 'Standard dev(%)', 
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
            
            pltr.plot_stats(self.stats, strats, items, **params)
        
        elif style=='simple':
            items = {
                'cagr': 'CAGR(%)', 
                'std': 'Standard dev(%)', 
                'sharpe': 'Sharpe', 
                'mdd': 'MDD(%)', 
            }          
        
            pltr.plot_stats(self.stats, strats, items, ncols=4, **params)
        
        
    def plot_profile(self, strats, **params):
        pltr.plot_profile(self.stats, strats, **params)
        
        
    def plot_dist(self, strats, **params):
        items = {
            ev._cagr: 'CAGR(%,Rolling1Y)', 
            ev._std: 'Standard dev(%,Rolling1Y)',
            ev._sharpe: 'Sharpe(Rolling1Y)', 
        }
        
        pltr.plot_dist(self.cum, strats, items, **params)
        
        
    def plot_contr_cum(self, **params):
        pltr.plot_contr_cum(self.model_contr, **params)
        
        
    def plot_breakdown(self):
        pltr.plot_breakdown(self.model_contr, self.weight)