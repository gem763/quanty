import numpy as np
import pandas as pd
import time
from collections import namedtuple, OrderedDict
from .plotter import Plotter as pltr
from ..model import evaluator as ev


            
class BacktesterBase(object):
    def __init__(self, params, **opt):
        params_flat = {}
        self.dict_flatten(params, params_flat)
        params_flat = self.overwrite_params(params_flat, **opt)
        self.__dict__.update(params_flat)
        
        dates, dates_asof = self._get_dates()
        p_close, p_buy, p_sell, r = self._prices()
        self.__dict__.update({
            'dates': dates, 
            'dates_asof': dates_asof, 
            'p_close': p_close, 
            'p_buy': p_buy, 
            'p_sell': p_sell, 
            'r': r, 
        })
        
        self.port = self.Port(dates_asof, **self.__dict__) 
        #self.port = DualMomentumPort(dates_asof, **self.__dict__) 
        #self.dm = dm(**self.__dict__)        
        #self.port = port(**self.__dict__)
        
        st = time.time()
        self._run()
        print(time.time()-st)
        
        self.turnover = ev._turnover(self.port.weight)
        self.stats = ev._stats(self.cum, self.beta_to, self.stats_n_roll)

        
    def _run(self):
        raise NotImplementedError

        
    def overwrite_params(self, base_params, **what):
        out = base_params.copy()
        out.update(what)
        return out        
        

    def dict_flatten(self, params, out):
        for k,v in params.items():
            if isinstance(v, dict):
                self.dict_flatten(v, out)
            else:
                out.update({k:v})        
                
        
    def _assets_all(self):
        assets_all = self.assets | {self.supporter, self.cash_equiv, self.beta_to}

        if self.bm is not None: assets_all.update({self.bm})
        if self.market is not None: assets_all.update({self.market})
        
        trade_assets = dict(self.trade_assets)

        if len(trade_assets)!=0:
            assets_all.update(set.union(*[set(trade_assets[k].keys()) for k in trade_assets.keys() if k in assets_all]))
        
        return assets_all        
        

    def _prices(self):
        assets_all = self._assets_all()
        db_unstacked = self.db.unstack().loc[:self.end].fillna(method='ffill')
        
        p_close = db_unstacked[self.price_src].reindex(columns=assets_all)
        p_high = db_unstacked['high'].reindex(columns=assets_all)
        p_low = db_unstacked['low'].reindex(columns=assets_all)
        
        p_close_high = p_close * (p_high/p_close).mean()
        p_close_low = p_close * (p_low/p_close).mean()

        p_close_high.update(p_high)
        p_close_low.update(p_low)
        
        if self.trade_tol=='at_close':
            p_buy = p_close
            p_sell = p_close
            
        elif self.trade_tol=='buyHigh_sellLow':
            p_buy = p_close_high
            p_sell = p_close_low
            
        elif self.trade_tol=='buyLow_sellHigh':
            p_buy = p_close_low
            p_sell = p_close_high
            
        return p_close, p_buy, p_sell, p_close.pct_change()    
               

    # 모든 영업일 출력
    def _get_dates(self):
        dates_all = self.db.index.levels[0]
        dates = dates_all[(self.start<=dates_all) & (dates_all<=self.end)]
        
        # 무조건 첫날(start)과 마지막날(end)은 포함
        if self.start not in dates: 
            dates = dates.insert(0, pd.Timestamp(self.start))
            
        if self.end not in dates: 
            dates = dates.append(pd.DatetimeIndex([self.end]))

        dates_asof = pd.date_range(self.start, self.end, freq='M')
        dates_asof = dates_all[dates_all.get_indexer(dates_asof, method='ffill')] & dates
        
        # 무조건 첫날(start)은 리밸 기준일
        if self.start not in dates_asof: 
            dates_asof = dates_asof.insert(0, pd.Timestamp(self.start))

        return dates, dates_asof
    
    
    def plot_cum(self, strats, **params):
        pltr.plot_cum(self.cum, strats, **params)
        
        
    def plot_cum_te(self, strats, bm, te_target, **params):
        pltr.plot_cum_te(self.cum, strats, bm, te_target, **params)
        
        
    def plot_cum_exc_te(self, strats, bm, te_target, **params):
        pltr.plot_cum_exc_te(self.cum, strats, bm, te_target, **params)
        
        
    def plot_cum_te_many(self, strats, **params):
        bm = list(self.backtests.values())[0].bm
        te_target_list = [bt.te_target for bt in self.backtests.values()]
        etas = [bt.eta for bt in self.backtests.values()]
        pltr.plot_cum_te_many(self.cum, strats, bm, te_target_list, etas, **params)
        
        
    def plot_cum_yearly(self, strats, **params): 
        pltr.plot_cum_yearly(self.cum[strats], **params)

        
    def plot_turnover(self):
        pltr.plot_turnover(self.turnover)
        
        
    def plot_weight(self, rng): 
        pltr.plot_weight(self.weight, rng, self.supporter, self.cash_equiv)
        
        
    def plot_stats(self, strats, style=None, **params):
        if style is None:
            items = {
                'cagr': 'CAGR (%)', 
                'std': 'Volatility (%)', 
                'sharpe': 'Sharpe', 
                'cagr_roll_med': 'CAGR (%,Rolling1Y)', 
                'std_roll_med': 'Volatility (%,Rolling1Y)', 
                'sharpe_roll_med': 'Sharpe (Rolling1Y)', 
                'mdd': 'MDD (%)', 
                'hit': 'Hit ratio (%,1M)', 
                'profit_to_loss': 'Profit-to-loss (%,1M)', 
                'beta': 'Beta (vs.' + self.beta_to + ')',
                'loss_proba': 'Loss probability (%,1Y)', 
                'consistency': 'Consistency (%)',
            }
            
            pltr.plot_stats(self.stats, strats, items, **params)
        
        elif style=='normal':
            items = {
                'cagr': 'CAGR (%)', 
                'std': 'Volatility (%)', 
                'sharpe': 'Sharpe', 
                'mdd': 'MDD (%)', 
                'hit': 'Hit ratio (%,1M)', 
                'profit_to_loss': 'Profit-to-loss (%,1M)', 
                'beta': 'Beta (vs.' + self.beta_to + ')',
                'loss_proba': 'Loss probability (%,1Y)', 
                'consistency': 'Consistency (%)',
            }
            
            pltr.plot_stats(self.stats, strats, items, **params)
            
        elif style=='simple':
            items = {
                'cagr': 'CAGR (%)', 
                'std': 'Volatility (%)', 
                'sharpe': 'Sharpe', 
                'mdd': 'MDD (%)', 
            }          
        
            pltr.plot_stats(self.stats, strats, items, ncols=4, **params)
        
        
    def plot_profile(self, strats, **params):
        pltr.plot_profile(self.stats, strats, **params)
        
        
    def plot_dist(self, strats, **params):
        items = {
            ev._cagr: 'CAGR (%,Rolling1Y)', 
            ev._std: 'Volatility (%,Rolling1Y)',
            ev._sharpe: 'Sharpe (Rolling1Y)', 
        }
        
        pltr.plot_dist(self.cum, strats, items, **params)
        
        
    def plot_contr_cum(self, **params):
        pltr.plot_contr_cum(self.model_contr, **params)
        
        
    def plot_breakdown(self):
        pltr.plot_breakdown(self.model_contr, self.weight)
        
        
        
        
        
class BacktestComparator(BacktesterBase):
    def __init__(self, params, **backtests):
        params_flat = {}
        self.dict_flatten(params, params_flat)
        self.__dict__.update(params_flat)
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
        self.stats = ev._stats(self.cum, self.beta_to, self.stats_n_roll)
        
        
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
        
        for v in tqdm_notebook(grid_values):
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