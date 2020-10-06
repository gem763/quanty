import numpy as np
import pandas as pd
from IPython.core.debugger import set_trace
from ..model import setting


def read_db(base='prices_global.pkl', add=[]):
    mapper = setting.mapper
    db = pd.read_pickle(base)
    #set_trace()
    if len(add)>0:
        for add_ in add:
            db_add = pd.read_pickle(add_)
            db_add = db_add.unstack().reindex(index=db.index.levels[0], method='ffill').stack()
            db = db.append(db_add)
    
    db = db.iloc[db.index.get_level_values(1).isin(mapper.keys())]
    return db.rename(index=mapper, level=1)