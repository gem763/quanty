import pandas as pd
import numpy as np
import itertools
import time
from pandas.tseries.offsets import Day
from IPython.core.debugger import set_trace
from tqdm import tqdm, tqdm_notebook

# Custom modules
from .dual_momentum import DualMomentumPort
from .backtester_base import BacktesterBase



class Backtester(BacktesterBase):
    
    def __init__(self, params, **opt):

        # 매일 기록
        self.book = []
        self.book_items = ['trade_amount', 'value', 'trade_cashflow', 'cost', 'cash', 'nav']
        self.i_trade_amount = self.book_items.index('trade_amount')
        self.i_value = self.book_items.index('value')
        self.i_trade_cashflow = self.book_items.index('trade_cashflow')
        self.i_cost = self.book_items.index('cost')
        self.i_cash = self.book_items.index('cash')
        self.i_nav = self.book_items.index('nav')
        self.book_items_n = len(self.book_items)
        
        # 리밸일에만 기록
        self.hold = []
        self.weight = []
        self.eq_value = []
        
        BacktesterBase.__init__(self, params, Port=DualMomentumPort, **opt)
        
            
    def _last_of(self, which, alt=None):
        try:
            return which[-1]
            
        except:
            return (None, alt)
        
          
    def _hold_last(self):
        return self._last_of(self.hold, alt=pd.Series())
        
                
    def _cash_last(self):
        date, book_ = self._last_of(self.book, alt=[])
        
        # len(book_)=0 인 경우는 없다. 
        # 이 함수가 불러질 때는, 적어도 book이 하나이상 채워졌을 때이다. 
        return date, book_[self.i_cash] 
        
        
    def _weight_last(self):
        return self._last_of(self.weight, alt=pd.Series())
        
        
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
    
    
    def _fill_book(self, date):
        
        if len(self.book)!=0:
            date_last = self.book[-1][0]
            dates_update = self.dates[(date_last<self.dates) & (self.dates<=date)]
            
            if len(dates_update)!=0:
                hold_last = self._hold_last()[1]
                cash_last = self._cash_last()[1]
            
                p_close = self.p_close[hold_last.index].reindex(dates_update, method='ffill')
                eq_value_update = p_close.mul(hold_last, axis=1)
                value_update = eq_value_update.sum(axis=1)

                book_update = np.zeros((len(dates_update), self.book_items_n))
                book_update[:,self.i_value] = value_update
                book_update[:,self.i_cash] = cash_last
                book_update[:,self.i_nav] = value_update + cash_last

                self.book += zip(dates_update, book_update.tolist())
                
        else:
            self.book.append((date, self._book(0, 0, 0, 0, self.cash)))
    
    
    
    def _df_of(self, which, columns=None):
        return pd.DataFrame.from_dict(dict(which), orient='index', columns=columns).fillna(0).sort_index()
    

    def _rebalance(self, date, weight_):
        # 이게 있으면, 리밸일마다 기록되는 것들(eq_value, hold 등)이 기록이 안되는 경우가 있다. 
        #if weight_.sub(self._weight_last()[1], fill_value=0).abs().sum()==0:
        #    return
        
        i_date_trade = self.dates.get_loc(date) + self.trade_delay
        
        if i_date_trade > len(self.dates)-1:
            return
        
        else:
            date_trade = self.dates[i_date_trade]
            
        #date_trade = self.dates[self.dates.get_loc(date) + self.trade_delay]

        if (date_trade in self.p_close.index):# & (date_trade <= pd.Timestamp(self.end)):
            self._fill_book(date_trade-Day())
            #set_trace()
            trade_amount_, trade_cashflow_, cost_, cash_, hold_ = self._trade(date_trade, weight_, self._hold_last()[1], self._cash_last()[1])
            eq_value_ = self._eq_value(date_trade, hold_)
            book_ = self._book(trade_amount_, eq_value_.sum(), trade_cashflow_, cost_, cash_)
            
            self.eq_value.append((date_trade, eq_value_))
            self.book.append((date_trade, book_))
            self.hold.append((date_trade, hold_))
            #self.weight.append((date_trade, weight_))
            

    def _positionize(self, date):
        self._fill_book(date)
        weight_ = self.port.portfolize(date, book=self.book)
        self.weight.append((date, weight_))
        return weight_
    
    
    def _cum(self):
        cum = self.p_close.reindex(self.dates, method='ffill')
        cum['DualMomentum'] = self.book['nav']
        return cum / cum.bfill().iloc[0]
    
    
    def _run(self):
        for date in tqdm_notebook(self.dates_asof):
            weight_ = self._positionize(date)
            self._rebalance(date, weight_)
        
        self._fill_book(self.end)
        self.book = self._df_of(self.book, columns=self.book_items)
        self.hold = self._df_of(self.hold)
        self.eq_value = self._df_of(self.eq_value)
        self.weight = self._df_of(self.weight)
        self.cum = self._cum()
