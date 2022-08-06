import datetime
import pandas as pd
import math
from scipy import stats, optimize
import numpy as np
import time
from datetime import timedelta
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
import warnings
from column_names import *


class OHCLNaNFillin:
    
    def __init__(self, df):
        self.df = df.copy()
        self.fill_in_columns = ['open', 'close', 'high', 'low']
        
        # verify essential columns
        if not set(self.fill_in_columns).issubset(self.df.columns):
            cols_str = ', '.join(self.fill_in_columns)
            warnings.warn(f'WARNING: {cols_str} must be in data columns, the missing column(s) will be initialized in by `np.nan`')
        for c in self.fill_in_columns:
            if c not in self.df.columns: self.df[c] = np.nan
    
    
    @staticmethod
    def row_fillin_open(row_OHCL):
        if np.isnan(row_OHCL['open']):
            if not np.isnan(row_OHCL['close']):
                return row_OHCL['close']
            else:
                return np.nanmean([row_OHCL['low'], row_OHCL['high']])
        return row_OHCL['open']
    
    @staticmethod
    def row_fillin_low(row_OHCL):
        if np.isnan(row_OHCL['low']):
            return np.nanmin(row_OHCL[:])
        return row_OHCL['low']
    
    @staticmethod
    def row_fillin_high(row_OHCL):
        if np.isnan(row_OHCL['high']):
            return np.nanmax(row_OHCL[:])
        return row_OHCL['high']

    @staticmethod
    def row_fillin_close(row_OHCL):
        if np.isnan(row_OHCL['close']):
            if not (np.isnan(row_OHCL['high']) or np.isnan(row_OHCL['low'])):
                return np.nanmean([row_OHCL['low'], row_OHCL['high']])
            else:
                if not np.isnan(row_OHCL['open']):
                    return row_OHCL['open']
                return np.nanmean([row_OHCL['low'], row_OHCL['high']])
        return row_OHCL['close']
    
    
    def run(self):
        df_out = self.df.copy()
        df_out.dropna(axis=0, how='all', inplace = True) # remove rows with no data
        
        for c in self.fill_in_columns:
            fillin_rule = getattr(self, f'row_fillin_{c}')
            df_out[c] = df_out.apply(lambda x: fillin_rule(x),axis=1)
        return df_out
    
    
    def test_run(self):
        # Test by creating a randomly filled-in NaN values:
        nan_mat = np.random.random(self.df.shape) < 0.2
        df_NaN = self.df.mask(nan_mat)
        df_NaN_start = df_NaN.copy()
        
        df_NaN.dropna(axis=0, how='all', inplace = True) # remove rows with no data
        
        for c in self.fill_in_columns:
            fillin_rule = getattr(self, f'row_fillin_{c}')
            df_NaN[c] = df_NaN.apply(lambda x: fillin_rule(x),axis=1)
        return df_NaN_start, df_NaN
