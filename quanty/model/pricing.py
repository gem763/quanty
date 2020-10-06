import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler#, Imputer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.pipeline import Pipeline as skpipe
from sklearn.tree import DecisionTreeClassifier



class PriceModeler(object):
    def __init__(self, prices_df, scaler=StandardScaler(), reducer=PCA(n_components=0.9)):
        self.p = prices_df
        self.log_p = np.log(prices_df)
        self.scaler = scaler
        self.reducer = reducer
        self.T, self.log_p_model = self._modeling()
        

    #@classmethod
    #def from_db(cls, asof, field='adj', n_bars=250, n_freq=1, univ='k200', scaler=StandardScaler(), reducer=PCA(n_components=0.9)):
    #    p = read_trailing_to(field, asof, n_bars, n_freq, univ).dropna(axis=1, how='any')
    #    return cls(p, scaler, reducer)
        
        
    def _modeling(self):
        log_p_scaled = self.scaler.fit_transform(self.log_p)
        T = self.reducer.fit_transform(log_p_scaled)
        log_p_model = self.scaler.inverse_transform(self.reducer.inverse_transform(T))
        # 참고: reducer.inverse_transform(T) = T.dot(reducer.components_)
        return T, pd.DataFrame(log_p_model, index=self.log_p.index, columns=self.log_p.columns)

    
    def plot_of(self, target, scale=None):
        if type(target) is str:
            target = self.p.columns.get_loc(target)

        symbol = self.p.columns[target]
        params = dict(legend=True, figsize=(8,5))
        
        if scale=='log':
            self.log_p.iloc[:,target].plot(color='r', label=symbol, **params)
            self.log_p_model.iloc[:,target].plot(color='k', label=str(symbol)+'(model)', **params)
            
        else:
            self.p.iloc[:,target].plot(color='r', label=symbol, **params)
            np.exp(self.log_p_model).iloc[:,target].plot(color='k', label=str(symbol)+'(model)', **params)
    
    
    def dislocation(self, method=None):
        if (method==None) or (method=='return'):
            return self.log_p - self.log_p_model
        
        elif method=='z':
            diff = self.dislocation()
            std = np.sqrt((diff**2).sum(axis=0) / len(diff))
            return diff / std
        
        elif method=='pct_rank':
            diff = self.dislocation()
            return diff.rank(axis=0, pct=True)
                    
        elif method=='direction':
            diff = self.dislocation()
            diff[diff>0] = 1
            diff[diff<0] = -1
            return diff
        
    
    def projection(self, n_proj, what='model', method='kalman'):
        if what=='model':
            return self.log_p_model.diff(n_proj)
        
        elif what=='market':
            return self.log_p.diff(n_proj)