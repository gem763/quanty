import pandas as pd
import numpy as np
import itertools
import time
from pandas.tseries.offsets import Day
from IPython.core.debugger import set_trace
#from numba import jit, float64, types
from tqdm import tqdm, tqdm_notebook

# Custom modules
from .plotter import Plotter as pltr
from .dual_momentum import DualMomentumPort
from .backtester_base import BacktesterBase



class Backtester(BacktesterBase):
    
    def __init__(self, params, **opt):
        BacktesterBase.__init__(self, params, Port=DualMomentumPort, **opt)
        
    
    def _begin_of_day(self):
        trade_amount_ = trade_cashflow_ = cost_ = 0
        
        try:
            hold_ = self.hold.iloc[-1].copy()
            cash_ = self.wealth.cash.iloc[-1]
            weight_ = self.weight.iloc[-1].copy()

        except:
            hold_ = pd.Series()
            cash_ = self.cash
            weight_ = pd.Series()

        return hold_, cash_, weight_, trade_amount_, trade_cashflow_, cost_

    
    def _end_of_day(self, date, hold_, cash_):
        if date==self.dates[0]:
            eq_value_ = pd.Series()
            
        else:
            eq_value_ = hold_ * self.p_close.loc[:date].iloc[-1] # 이게 더 빠름
            #eq_value_ = hold_ * self.p_close.reindex([date], method='ffill').iloc[0]

        value_ = eq_value_.sum()
        nav_ = value_ + cash_
        return eq_value_, value_, nav_    
    
    
    
    def _run(self):
        trade_due = -1
        cols = self.port.weight.columns
        
        # 매일 기록
        self.hold = pd.DataFrame(columns=cols, dtype=float)
        self.eq_value = pd.DataFrame(columns=cols, dtype=float)
        self.wealth = pd.DataFrame(columns=['trade_amount', 'value', 'trade_cashflow', 'cost', 'cash', 'nav'], dtype=float)
        
        # 의사결정일에만 기록
        self.weight = pd.DataFrame(columns=cols, dtype=float)        
        
                
        for date in tqdm_notebook(self.dates):
            if date in self.p_close.index: 
                trade_due -= 1
                
            # Begin of the day
            hold_, cash_, weight_, trade_amount_, trade_cashflow_, cost_ = self._begin_of_day()
                        
            # 0. 리밸런싱 실행하는 날
            if trade_due==0:
                #if date>pd.Timestamp('2002-12-31'): set_trace()
                trade_amount_, trade_cashflow_, cost_, cash_, hold_ = self._rebalance(date, hold_, cash_, weight_)

            # 1. 리밸런싱 비중결정하는 날
            elif date in self.dates_asof:
                weight_, trade_due = self._positionize(date, weight_, trade_due)                
                self.weight.loc[date] = weight_
                #self.weight.append(weight_)
              
            # 2. 아무일도 없는 날
            else:
                pass
                
            # End of the day
            eq_value_, value_, nav_ = self._end_of_day(date, hold_, cash_)
            
            self.hold.loc[date] = hold_  #.append(hold_)
            self.eq_value.loc[date] = eq_value_  #.append(eq_value_)
            self.wealth.loc[date]= [trade_amount_, value_, trade_cashflow_, cost_, cash_, nav_]
            #.append([trade_amount_, value_, trade_cashflow_, cost_, cash_, nav_])


        # 종목별 시그널, 포지션
        #self.weight = pd.DataFrame(self.weight, index=self.dates_asof)
        
        # Daily Booking
        #self.hold = pd.DataFrame(self.hold, index=self.dates)
        #self.eq_value = pd.DataFrame(self.eq_value, index=self.dates)
        #self.wealth = pd.DataFrame(self.wealth, index=self.dates, columns=['trade_amount', 'value', 'trade_cashflow', 'cost', 'cash', 'nav'])
        
        # 지수가격(normalized)
        cum = self.p_close.reindex(self.dates, method='ffill')
        cum['DualMomentum'] = self.wealth['nav']
        self.cum = cum / cum.bfill().iloc[0]



    def _rebalance(self, date, hold_, cash_, pos_tobe_):
        if self.trade_prev_nav_based:
            pos_prev_amount = hold_*self.p_close.loc[:date-Day()].iloc[-1]
        else:
            pos_prev_amount = hold_*self.p_close.loc[:date].iloc[-1]
            
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
        #set_trace()
        weight_ = self.port.weight.loc[date]
        
        if weight_.sub(weight_asis_, fill_value=0).abs().sum()!=0:
            trade_due = self.trade_delay

        return weight_, trade_due

                

        
class Backtester2(BacktesterBase):
    
    def __init__(self, params, **opt):

        # 매일 기록
        self.hold = []
        self.eq_value = []
        self.wealth = []
        
        # 의사결정일에만 기록
        self.weight = []

        BacktesterBase.__init__(self, params, Port=DualMomentumPort, **opt)
        
    
    def _begin_of_day(self):
        trade_amount_ = trade_cashflow_ = cost_ = 0
        
        try:
            hold_ = self.hold[-1].copy()
            cash_ = self.wealth[-1][-2]
            weight_ = self.weight[-1].copy()

        except:
            hold_ = pd.Series()
            cash_ = self.cash
            weight_ = pd.Series()

        return hold_, cash_, weight_, trade_amount_, trade_cashflow_, cost_

    
    def _end_of_day(self, date, hold_, cash_):
        if date==self.dates[0]:
            eq_value_ = pd.Series()
            
        else:
            eq_value_ = hold_ * self.p_close.loc[:date].iloc[-1] # 이게 더 빠름
            #eq_value_ = hold_ * self.p_close.reindex([date], method='ffill').iloc[0]

        value_ = eq_value_.sum()
        nav_ = value_ + cash_
        return eq_value_, value_, nav_    
    
    
    
    def _run(self):
        trade_due = -1
                
        for date in tqdm_notebook(self.dates):
            if date in self.p_close.index: 
                trade_due -= 1
                
            # Begin of the day
            hold_, cash_, weight_, trade_amount_, trade_cashflow_, cost_ = self._begin_of_day()
                        
            # 0. 리밸런싱 실행하는 날
            if trade_due==0:
                #if date>pd.Timestamp('2002-12-31'): set_trace()
                trade_amount_, trade_cashflow_, cost_, cash_, hold_ = self._rebalance(date, hold_, cash_, weight_)

            # 1. 리밸런싱 비중결정하는 날
            elif date in self.dates_asof:
                weight_, trade_due = self._positionize(date, weight_, trade_due)                
                self.weight.append(weight_)
              
            # 2. 아무일도 없는 날
            else:
                pass
                
            # End of the day
            eq_value_, value_, nav_ = self._end_of_day(date, hold_, cash_)
            
            self.hold.append(hold_)
            self.eq_value.append(eq_value_)
            self.wealth.append([trade_amount_, value_, trade_cashflow_, cost_, cash_, nav_])


        # 종목별 시그널, 포지션
        self.weight = pd.DataFrame(self.weight, index=self.dates_asof)
        
        # Daily Booking
        self.hold = pd.DataFrame(self.hold, index=self.dates)
        self.eq_value = pd.DataFrame(self.eq_value, index=self.dates)
        self.wealth = pd.DataFrame(self.wealth, index=self.dates, columns=['trade_amount', 'value', 'trade_cashflow', 'cost', 'cash', 'nav'])
        
        # 지수가격(normalized)
        cum = self.p_close.reindex(self.dates, method='ffill')
        cum['DualMomentum'] = self.wealth['nav']
        self.cum = cum / cum.bfill().iloc[0]



    def _rebalance(self, date, hold_, cash_, pos_tobe_):
        if self.trade_prev_nav_based:
            pos_prev_amount = hold_*self.p_close.loc[:date-Day()].iloc[-1]
        else:
            pos_prev_amount = hold_*self.p_close.loc[:date].iloc[-1]
            
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
        weight_ = self.port.weight.loc[date]
        
        if weight_.sub(weight_asis_, fill_value=0).abs().sum()!=0:
            trade_due = self.trade_delay

        return weight_, trade_due        