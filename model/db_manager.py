import numpy as np
import pandas as pd
from ..model import setting


def read_db(base='prices_global.pkl', add=None):
    mapper = setting.mapper
    db = pd.read_pickle(base)
    
    if add:
        db_add = pd.read_pickle(add)
        db_add = db_add.unstack().reindex(index=db.index.levels[0], method='ffill').stack()
        db = db.append(db_add)
    
    db = db.iloc[db.index.get_level_values(1).isin(mapper.keys())]
    return db.rename(index=mapper, level=1)