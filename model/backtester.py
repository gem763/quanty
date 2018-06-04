import pandas as pd
import numpy as np
import itertools
import time
from pandas.tseries.offsets import Day
from IPython.core.debugger import set_trace
from collections import namedtuple, OrderedDict
from numba import jit, float64, types
from tqdm import tqdm

# Custom modules
from .plotter import Plotter as pltr
from .dual_momentum import DualMomentum as dm
from .portfolizer import Portfolio as port
from ..model import evaluator as ev



@jit(types.Tuple((float64[:], float64, float64[:]))(float64[:], float64[:]), nopython=True)
def _update_pos_daily_fast(pos_daily_last_np, r):
    pos_cash = 1.0 - pos_daily_last_np.sum()
    pos_updated = (1+r) * pos_daily_last_np
    pos_total = pos_cash + pos_updated.sum()
    if pos_total==0: pos_total = 1.0
    model_rtn_ = pos_total - 1.0
    model_contr_ = pos_updated - pos_daily_last_np

    return pos_updated/pos_total, model_rtn_, model_contr_



class BacktesterBase(object):
    def __init__(self, params, **opt):
        params = self._overwrite_params(params, **opt)
        
        # 변수 초기화
        self.__dict__.update(params)
        self.dates, self.dates_asof = self._get_dates()
        self.p, self.p_ref, self.p_close, self.p_buy, self.p_sell, self.r = self._prices()
        self.dm = dm(**params, p_ref=self.p_ref, p_close=self.p_close, dates_asof=self.dates_asof)
        self.port = port(self.w_type, self.cash_equiv, self.p_close, self.iv_period, self.apply_kelly)
        
        
        # 백테스트
#        st=time.time()
        self._run()
#        print(time.time()-st)
        
#        st=time.time()
        self.turnover = ev._turnover(self.weight)
#        print(time.time()-st)
        
#        st=time.time()
        self.stats = ev._stats(self.cum, self.beta_to, self.n_roll_stats)
#        print(time.time()-st)

        
    def _run(self):
        raise NotImplementedError
        
                
    def _prices(self):    
        data_unstacked = self.data.unstack().loc[:self.end].fillna(method='ffill')
        
        assets = set(self.assets_member.values.flatten())
        assets.update({self.beta_to, self.cash_equiv})
        
        assets_bet = set(self.assets_member.bet)
        assets_bet.update({self.cash_equiv})
        
        p = data_unstacked[self.perf_src].reindex(columns=assets)
        p_ref = data_unstacked[self.ref_src].reindex(columns=self.assets_member.ref)
        p_bet = data_unstacked[self.bet_src].reindex(columns=assets_bet)
        p_high = data_unstacked['high'].reindex(columns=assets_bet)
        p_low = data_unstacked['low'].reindex(columns=assets_bet)
        
        p_bet_high = p_bet * (p_high/p_bet).mean()
        p_bet_low = p_bet * (p_low/p_bet).mean()

        p_bet_high.update(p_high)
        p_bet_low.update(p_low)
        
        if self.trading_tolerance=='at_close':
            p_close = p_bet
            p_buy = p_bet
            p_sell = p_bet
            
        elif self.trading_tolerance=='buyHigh_sellLow':
            p_close = p_bet            
            p_buy = p_bet_high
            p_sell = p_bet_low
            
        elif self.trading_tolerance=='buyLow_sellHigh':
            p_close = p_bet
            p_buy = p_bet_low
            p_sell = p_bet_high
            
        return p, p_ref, p_close, p_buy, p_sell, p.pct_change()    
            
            
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
            dates = dates.append(pd.DatetimeIndex([self.end]))

        #dates_ref = pd.date_range(dates_all[0], self.end, freq='M')
        #set_trace()
        dates_asof = pd.date_range(self.start, self.end, freq='M')
        dates_asof = dates_all[dates_all.get_indexer(dates_asof, method='ffill')] & dates
        
        # 무조건 첫날(start)은 리밸 기준일
        if self.start not in dates_asof: 
            dates_asof = dates_asof.insert(0, pd.Timestamp(self.start))

        return dates, dates_asof
    
    
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
        
        elif style=='normal':
            items = {
                'cagr': 'CAGR(%)', 
                'std': 'Standard dev(%)', 
                'sharpe': 'Sharpe', 
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
        
        
    

class Backtester(BacktesterBase):
    def init_of_the_day(self):
        trade_amount_ = trade_cashflow_ = cost_ = 0
        
        try:
            hold_ = self.hold[-1].copy()
            cash_ = self.wealth[-1][-2]
            weight_ = self.weight[-1].copy()
            pos_ = self.pos[-1].copy()
            pos_d_ = self.pos_d[-1].copy()

        except:
            hold_ = pd.Series()
            cash_ = self.cash
            weight_ = pd.Series()
            pos_ = pd.Series()
            pos_d_ = pd.Series()

        return hold_, cash_, weight_, pos_, pos_d_, trade_amount_, trade_cashflow_, cost_
        
    
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
        self.selection = []
        
        for date in tqdm(self.dates):
            if date in self.p.index: 
                trade_due -= 1
                
            # Initialize for today
            hold_, cash_, weight_, pos_, pos_d_, trade_amount_, trade_cashflow_, cost_ = self.init_of_the_day()
            
            
            # 0. 리밸런싱 실행하는 날
            if trade_due==0:
                trade_amount_, trade_cashflow_, cost_, cash_, hold_ = self._rebalance(date, hold_, cash_, weight_)
                pos_d_, model_rtn_, model_contr_ = self._update_pos_daily(date, pos_d_)
                pos_d_ = pos_.copy()
                

            # 1. 리밸런싱 비중결정하는 날
            elif date in self.dates_asof:
                weight_, pos_, trade_due, kelly_output = self._positionize(date, weight_, trade_due)
                pos_d_, model_rtn_, model_contr_ = self._update_pos_daily(date, pos_d_)
                
                #self.sig.append(sig_)
                #self.ranks.append(ranks_)
                self.weight.append(weight_)
                self.pos.append(pos_)
                self.kelly.append(kelly_output)
                #self.selection.append(selection_)
                
              
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
        self.selection = pd.DataFrame(self.selection, index=self.dates_asof)
        
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
            
            
    def _update_pos_daily(self, date, pos_daily_last_):    
        if date in self.r.index:
            assets = pos_daily_last_.index[pos_daily_last_!=0]
            pos_daily_last_np = pos_daily_last_.loc[assets].values
            r = self.r.loc[date, assets].values
            pos_daily_last_np, model_rtn_, model_contr_ = _update_pos_daily_fast(pos_daily_last_np, r)
            
            return pd.Series(pos_daily_last_np, index=assets), model_rtn_, model_contr_
        
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
        selection_, sig_, ranks_ = self.dm.selection.loc[date], self.dm.sig.loc[date], self.dm.ranks.loc[date]
        weight_, pos_, kelly_output = self.port.get(selection_, date, sig_, ranks_, self.wealth, self.model_rtn)
        
        if weight_.sub(weight_asis_, fill_value=0).abs().sum()!=0:
            trade_due = self.trade_delay

        return weight_, pos_, trade_due, kelly_output


    def _evaluate(self, date, hold_, cash_):
        if date==self.dates[0]:
            eq_value_ = pd.Series()
            
        else:
            eq_value_ = hold_ * self.p_close.loc[:date].iloc[-1]

        value_ = eq_value_.sum()
        nav_ = value_ + cash_
        return eq_value_, value_, nav_

                
        
class BacktestComparator(Backtester):
    def __init__(self, params, **backtests):
        self.__dict__.update(params)
        self.backtests = OrderedDict(backtests)
        self.cum, self.stats = self._get_results()


    def _get_results(self):
        for i, (k,v) in enumerate(self.backtests.items()):
            if i==0:
                cum = v.cum.copy()
                cum.rename(columns={'DualMomentum':k}, inplace=True)
              
                stats = v.stats.copy()
                stats.rename(index={'DualMomentum':k}, inplace=True)
                
            else:
                cum.loc[:,k] = v.cum.loc[:,'DualMomentum']
                stats.loc[k] = v.stats.loc['DualMomentum']
                
        return cum, stats
        
        
    def mix(self):
        self.cum = self.cum.fillna(method='ffill')
        r_mix = self.cum[list(self.backtests.keys())].pct_change()
        std_mix = r_mix.ewm(halflife=250, min_periods=20).std()
        
        dates_asof = list(self.backtests.values())[0].dates_asof
        alloc = 1.0 / std_mix.loc[dates_asof]
        alloc = alloc.div(alloc.sum(axis=1), axis=0).fillna(0.25)
        
        mixed = []
        
        for i_date, date in enumerate(self.cum.index):
            
            if i_date==0:
                cum_mix_ = alloc.loc[date]
                
            elif date in alloc.index:
                cum_mix_ = mixed[i_date-1] * (1+r_mix.loc[date])
                cum_mix_ = alloc.loc[date] * cum_mix_.sum()
                
            else:
                cum_mix_ = mixed[i_date-1] * (1+r_mix.loc[date])
                
            cum_mix_['sum'] = cum_mix_.sum()
            mixed.append(cum_mix_)
            
        mixed = pd.DataFrame(mixed, index=self.cum.index)
        self.cum['mixed'] = mixed['sum']
        self.stats = ev._stats(self.cum, self.beta_to, self.n_roll_stats)
        
        
    def plot_stats_pool(self, **params):
        items = {
            'cagr': 'CAGR(%)', 
            'std': 'Standard dev(%)', 
            'sharpe': 'Sharpe', 
            'mdd': 'MDD(%)',
        }
        
        pltr.plot_stats_pool(self.stats.loc[self.backtests.keys()], items, **params)
        
                
    @classmethod
    def compare(cls, base_params, **grids):
        params = base_params.copy()
        grid_keys = grids.keys()
        grid_values = list(itertools.product(*grids.values()))
        backtests = OrderedDict()
        bt = namedtuple('bt', grid_keys)
        
        for v in tqdm(grid_values):
            k = dict(zip(grid_keys, v))
            params.update(k)
            backtests[str(bt(**k))] = Backtester(params)
         
        return cls(params, **backtests)
      
      
    @classmethod
    def compare_highlow(cls, params):
        return cls.compare(params, trading_tolerance=['buyLow_sellHigh', 'at_close', 'buyHigh_sellLow'])
      

    def plot_cum_highlow(self, strats, **params):
        pltr.plot_cum(self.cum, strats, **params)
        plt.fill_between(self.cum.index, 
                         self.cum["bt(trading_tolerance='buyHigh_sellLow')"], 
                         self.cum["bt(trading_tolerance='buyLow_sellHigh')"], 
                         color=params['color'][-1], 
                         alpha=0.4)
