import pandas as pd
import numpy as np
import itertools
import time
from pandas.tseries.offsets import Day
from tqdm import tqdm, tqdm_notebook
from IPython.core.debugger import set_trace
from IPython import get_ipython

# Custom modules
from .dual_momentum import DualMomentumPort
from .backtester_base import BacktesterBase
from ..model import evaluator as ev


if 'zmqshell' in str(type(get_ipython())):
    prg = tqdm_notebook
    
else:
    prg = tqdm


class Backtester(BacktesterBase):
    
    def __init__(self, params, **opt):

        # 매일 기록
        self.p_max = []
        self.p_profitake = []
        self.r_losscut = []
        
        self.book = []
        self.book_items = ['trade_amount', 'value', 'trade_cashflow', 'cost', 'cash', 'nav']
        self.i_trade_amount = self.book_items.index('trade_amount')
        self.i_value = self.book_items.index('value')
        self.i_trade_cashflow = self.book_items.index('trade_cashflow')
        self.i_cost = self.book_items.index('cost')
        self.i_cash = self.book_items.index('cash')
        self.i_nav = self.book_items.index('nav')
        self.book_items_n = len(self.book_items)
        
        
        self.hold = []
        #self.eq_value = []
        
        # 리밸기준일(date_asof)일에만 기록
        self.weight = []
        
        BacktesterBase.__init__(self, params, Port=DualMomentumPort, **opt)
        
            
    def _last_of(self, which, alt=None):
        try:
            return which[-1]
            
        except:
            return (None, alt)
        
          
    def _p_max_last(self):
        return self._last_of(self.p_max, alt=pd.Series())
            
        
    def _r_losscut_last(self):    
        return self._last_of(self.r_losscut, alt=pd.Series())

    
    def _p_profitake_last(self):    
        return self._last_of(self.p_profitake, alt=pd.Series())

                
    def _hold_last(self):
        return self._last_of(self.hold, alt=pd.Series())
        
                
    def _cash_last(self):
        date, book_ = self._last_of(self.book, alt=[])
        
        # len(book_)=0 인 경우는 없다. 
        # 이 함수가 불러질 때는, 적어도 book이 하나이상 채워졌을 때이다. 
        return date, book_[self.i_cash] 
        
        
    def _book(self, trade_amount_, value_, trade_cashflow_, cost_, cash_):
        nav_ = value_ + cash_
        
        book_ = [0]*self.book_items_n
        book_[self.i_trade_amount] = trade_amount_
        book_[self.i_value] = value_
        book_[self.i_trade_cashflow] = trade_cashflow_
        book_[self.i_cost] = cost_
        book_[self.i_cash] = cash_
        book_[self.i_nav] = nav_
        return book_    

    
    def _threshold(self, p_close):
        p_max_last = self._p_max_last()[1]
        r_losscut_last = self._r_losscut_last()[1]
        p_profitake = self._p_profitake_last()[1]
        
        p_max_ = p_max_last.to_frame().T.append(p_close).cummax().iloc[1:]
        self.p_max += [(date_, p_max_.loc[date_]) for date_ in p_max_.index]
        #print(p_max_.columns)
        #sigma = r_losscut_last[p_max_.columns]
        sigma = r_losscut_last.reindex(p_max_.columns)
        
        p_losscut = p_max_*(1-sigma)
        p_rentry = p_max_*(1-self.rentry_sigma*sigma)        
        return p_losscut, p_rentry, p_profitake


    def _hold(self, p_close_, hold_last):
        p_losscut_, p_rentry_, p_profitake_ = self._threshold(p_close_)

        hold_ = p_close_.copy()
        hold_[:] = np.nan
        #hold_[p_thres1<p_close_] = 1.0
        hold_[(p_rentry_<p_close_) & (p_close_<=p_losscut_)] = 0.5
        hold_[p_close_<=p_rentry_] = 1.0
        #hold_[(p_rentry_<p_close_) & (p_close_<=p_losscut_)] = 0.5
        hold_[p_profitake_<=p_close_] = 0.0

        hold_ = hold_.fillna(method='ffill').fillna(1.0).shift()
        hold_ = hold_.mul(hold_last, axis=1)
        hold_.iloc[0] = hold_last
        return hold_
    
    
    def _trade_cashflow_by_asset(self, p_close_, hold_):
        trade_share = hold_.diff()
        trade_share.iloc[0] = 0
        return -trade_share * p_close_    
    

    def _fill_book(self, date):

        if len(self.book)!=0:
            date_last = self.book[-1][0]
            dates_update = self.dates[(date_last<self.dates) & (self.dates<=date)]

            if len(dates_update)!=0:
                hold_last = self._hold_last()[1]
                cash_last = self._cash_last()[1]

                p_close_ = self.p_close[hold_last.index].reindex(dates_update, method='ffill')
                hold_ = self._hold(p_close_, hold_last)
                trade_cashflow_by_asset = self._trade_cashflow_by_asset(p_close_, hold_)
                trade_cashflow_ = trade_cashflow_by_asset.sum(axis=1)
                trade_amount_ = abs(trade_cashflow_by_asset).sum(axis=1)
                cost_ = trade_amount_ * self.expense
                cash_ = (trade_cashflow_-cost_).cumsum() + cash_last
                eq_value_ = p_close_ * hold_
                value_ = eq_value_.sum(axis=1)

                book_update = np.zeros((len(dates_update), self.book_items_n))
                book_update[:,self.i_trade_amount] = trade_amount_
                book_update[:,self.i_value] = value_
                book_update[:,self.i_trade_cashflow] = trade_cashflow_
                book_update[:,self.i_cost] = cost_
                book_update[:,self.i_cash] = cash_
                book_update[:,self.i_nav] = value_ + cash_

                self.book += zip(dates_update, book_update.tolist())
                self.hold += [(date_, hold_.loc[date_]) for date_ in hold_.index]

        else:
            self.book.append((date, self._book(0, 0, 0, 0, self.cash)))



    def _df_of(self, which, columns=None):
        return pd.DataFrame.from_dict(dict(which), orient='index', columns=columns).fillna(0).sort_index()


    def _tolerance(self, date, weight_):
        n_days = int(len(self.dates)/len(self.dates_asof))
        weight__ = weight_[weight_!=0]
        cov = self.r[weight__.index].iloc[-250:].cov() * n_days
        std = weight__.T.dot(cov.dot(weight__))**0.5
        
        x = self.losscut / std
        vol = pd.Series(np.diag(cov)**0.5, index=weight__.index)
        r_losscut_ = x * vol
        r_profitake_ = self.profitake_sigma * vol
        return r_losscut_, r_profitake_
    
    
    def _rebalance(self, date, weight_):
        # 이게 있으면, 리밸일마다 기록되는 것들(eq_value, hold 등)이 기록이 안되는 경우가 있다. 
        #if weight_.sub(self._weight_last()[1], fill_value=0).abs().sum()==0:
        #    return

        i_date_trade = self.dates.get_loc(date) + self.trade_delay
        
        if i_date_trade > len(self.dates)-1:
            return
        
        else:
            date_trade = self.dates[i_date_trade]


        if date_trade in self.p_close.index:
            self._fill_book(date_trade-Day())
            trade_amount_, trade_cashflow_, cost_, cash_, hold_ = self._trade(date_trade, weight_, self._hold_last()[1], self._cash_last()[1])
            eq_value_, p_close = self._eq_value(date_trade, hold_)
            book_ = self._book(trade_amount_, eq_value_.sum(), trade_cashflow_, cost_, cash_)
            r_losscut_, r_profitake_ = self._tolerance(date, weight_)

            #self.eq_value.append((date_trade, eq_value_))
            self.book.append((date_trade, book_))
            self.hold.append((date_trade, hold_))
            
            self.p_max.append((date_trade, p_close))
            self.r_losscut.append((date_trade, r_losscut_))
            self.p_profitake.append((date_trade, p_close*(1+r_profitake_)))


    def _positionize(self, date):
        self._fill_book(date)
        weight_ = self.port.portfolize(date, book=self.book)
        #weight_ = weight_[weight_!=0]
        self.weight.append((date, weight_))
        return weight_
    
    
    def _cum(self):
        cum = self.p_close.reindex(self.dates, method='ffill')
        cum['DualMomentum'] = self.book['nav']
        return cum / cum.bfill().iloc[0]
    
    
    def _run(self):
        for date in prg(self.dates_asof):
            #if date==pd.Timestamp('2008-10-31'): set_trace()
            weight_ = self._positionize(date)
            self._rebalance(date, weight_)
        
        self._fill_book(self.end)
        self.book = self._df_of(self.book, columns=self.book_items)
        self.hold = self._df_of(self.hold)
        #self.eq_value = self._df_of(self.eq_value)
        self.weight = self._df_of(self.weight)
        self.p_max = self._df_of(self.p_max)
        #self.vol = self._df_of(self.vol)
        self.cum = self._cum()

        
        
class Book(object):
    def __init__(self):
        pass