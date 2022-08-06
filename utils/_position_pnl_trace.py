import datetime
import pandas as pd
import math
from scipy import stats, optimize
import numpy as np
import time
from tqdm import tqdm
from datetime import timedelta
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
import warnings

from column_names import *


class PositionPnLBookIntraday:
    '''
    Compute types of daily position and pnl columns from intraday trading data
    
    * Position:
        Open position: overnight position left from previous trading day
        Tally position: current trading day's cumulated position
        Net position = Open position + Tally position
    
    * Pnl: dollar[$] or percentage[%]
        Open pnl: pnl resulted from open position
        Tally pnl: pnl resulted from tally position
        Cost pnl: pnl resulted from trades (i.e., transaction costs)
        Net pnl = Open pnl + Tally pnl + Cost pnl
        
    Calculation formulas for PnL (at some time `t` at day `d+1`):
        open_pnl(t) = open_position * [p(t) - pc_d]; where pc_d is the closed price at day d
        tally_pnl(t) = sum of { [p(t) - tally_trade_price] * tally_trade_size , tally trades up to time t}
        cost_pnl(t) = sum of trade_cost up to time t
        net_pnl(t) = sum of above three
        

    * A method for calculating derivatives-trading PnL in percentage.
    Note that, in classical term, PnL (in perct) should be:
                final_return / initial_investment;
    from day-to-day view, the PnL for today's trade should be:
                return_of_today / end_of_yesterday_portfolio_valuation;
    
    However, for derivatives, they are all contracts, it's not very clear on how to compute PnL in percentage.
    Here, we do it in this way:
            net_pnl_today[$] / (open_trade_valuation[$] + open_position_valuation[$])
    This definition need to be revised and modified for more general cases
    '''
    
    def __init__(self, data: pd.DataFrame,
                 cost_rate: int = 0,
                 day_grouping: Union[str, List[Callable]] = None):
        '''
        Initialize input data
        Parameters
        ----------
        data : pd.DataFrame
            Trading size and price, noncumulative.
             - Time series with at least three columsn: trade_size, trade_price and close_price at each time tick
             - Example::
             
                time                close    trade_price    trade_size
                2015-07-16 09:10     100         100           0
                2015-07-16 09:15     123         123           -1
                2015-07-16 09:19     122         122           0
                2015-07-16 09:22     111         111           1
                
        cost_rate : float, optional
           The cost of making each buy or sell, pay fee: abs(cost_rate * trade_price * trade_size)
           
        day_grouping : whatever data format fits in groupby() in pandas
        '''
        
        if not isinstance(data, pd.DataFrame):
            raise TypeError("`data` must be a pandas.DataFrame with columns")

        # verify essential columns
        source_columns = [TRADE_SIZE, TRADE_PRICE, 'close']
        if not set(source_columns).issubset(data.columns):
            cols_str = ', '.join(source_columns)
            raise TypeError(f'{cols_str} must be in data input')
                
        # TODO: could add more check conditions for input attributes
        self._data = data.copy()
        self._cost_rate = cost_rate
        self._grouping = day_grouping

        
    
    def initialize_pnl_position_columns(self):
        '''
        initialize all the columns
        '''
        
        # numerical
        columns_to_initialize = [
            COST_TRADE, OPEN_PNL, TALLY_PNL, NET_PNL, COST_PNL,
            OPEN_PNL_PERCT, TALLY_PNL_PERCT, NET_PNL_PERCT,
            IS_OPEN_TRADE, OPENTRADE_SIZE,
            OPENTRADE_VALUATION, OPENPOSITION_VALUATION,
            OPEN_POSITION, TALLY_POSITION, NET_POSITION,
        ]
        for col in columns_to_initialize: self._data[col] = 0.0
        
        # categorical
        self._data[TRADE_TYPE] = 'no trade'
    
    
    
    ##################################################################
    ##################     helper  private functions   ###############
    ##################################################################
    def __position_tally(self, intraday_trade_series):
        return intraday_trade_series[TRADE_SIZE].cumsum().values
    
    def __position_after_close(self, intraday_trade_series):
        return intraday_trade_series[TRADE_SIZE].sum()
    
    def __open_close_trade_type(self, row):
        if abs(row[TRADE_SIZE]) < 1e-8: return 'no trade'
        pos_before_trade = row[NET_POSITION] - row[TRADE_SIZE] # net position before making current trade
        if (row[TRADE_SIZE] * pos_before_trade < -1e-8) and (row[NET_POSITION] * pos_before_trade > -1e-8):
            return 'close trade'
        return 'open trade'
    
    def __opentrade_size(self, row):
        if row[TRADE_TYPE] != 'open trade': return 0.
        if abs(row[TRADE_SIZE]) - abs(row[NET_POSITION]) > 1e-8:
            return row[NET_POSITION]
        return row[TRADE_SIZE]
                
    def __pnl_tally(self, intraday_trade_series):
        cum_pos = intraday_trade_series[TRADE_SIZE].cumsum().values
        cum_prod_of_size_price = (intraday_trade_series[TRADE_SIZE] * intraday_trade_series[TRADE_PRICE]).cumsum().values
        return (intraday_trade_series['close'] * cum_pos - cum_prod_of_size_price)
    
    def __pnl_open(self, intraday_trade_series, last_day_close_price):
        return ((intraday_trade_series['close'] - last_day_close_price) * intraday_trade_series[OPEN_POSITION]).values
    
    def __calculate_pnl_open_perct(self, row):
        if abs(row[OPEN_PNL]) < 1e-8:
            return 0.0
        return row[OPEN_PNL] / row[OPENPOSITION_VALUATION]
    
    def __calculate_pnl_tally_perct(self, row):
        if abs(row[TALLY_PNL]) < 1e-8:
            return 0.0
        try:
            row[TALLY_PNL] / row[OPENTRADE_VALUATION]
        except ZeroDivisionError:
            raise ZeroDivisionError(f'zero division error TODO')
            
        return row[TALLY_PNL] / row[OPENTRADE_VALUATION]
    
    def __calculate_pnl_net_perct(self, row):
        if abs(row[NET_PNL]) < 1e-8:
            return 0.0
        return row[NET_PNL] / (row[OPENPOSITION_VALUATION] + row[OPENTRADE_VALUATION])
    
    
    ##################################################################
    ##################     core run functions           ##############
    ##################################################################
    def add_transaction_costs(self):
        self._data[COST_TRADE] = -self._cost_rate * abs(self._data[TRADE_PRICE] * self._data[TRADE_SIZE])
        
    def run_position_trace(self):
        df_day_groups = self._data.groupby(self._grouping)
        open_position_for_nextday = None
        
        # iterate over everyday's trade trace
        for i, g in tqdm(df_day_groups, desc = 'position trace processing'):
            time.sleep(0.0001)
            
            # tally position
            self._data.loc[g.index, TALLY_POSITION] = self.__position_tally(g)
            
            # open position
            if open_position_for_nextday != None:
                self._data.loc[g.index, OPEN_POSITION] = open_position_for_nextday
            open_position_for_nextday = self.__position_after_close(g)

        self._data[NET_POSITION] = self._data[OPEN_POSITION] + self._data[TALLY_POSITION]
        
        # classify open and close trade
        self._data[TRADE_TYPE] = self._data.apply(lambda x: self.__open_close_trade_type(x), axis = 1)
        self._data[IS_OPEN_TRADE] = self._data.apply(lambda x: 1 if x[TRADE_TYPE] == 'open trade' else 0, axis = 1)
        self._data[OPENTRADE_SIZE] = self._data.apply(lambda x: self.__opentrade_size(x), axis = 1)
        
    
    def run_pnl_trace(self):
        df_day_groups = self._data.groupby(self._grouping)
        last_day_close_price = None
        
        # iterate over everyday's trade trace
        for i, g in tqdm(df_day_groups, desc = 'pnl trace processing'):
            time.sleep(0.0001)
            
            # tally pnl
            self._data.loc[g.index, TALLY_PNL] = self.__pnl_tally(g)
            
            # cost pnl
            self._data.loc[g.index, COST_PNL] = g[COST_TRADE].cumsum().values
            
            # open pnl
            if last_day_close_price != None:
                self._data.loc[g.index, OPEN_PNL] = self.__pnl_open(g, last_day_close_price)
            last_day_close_price = g.iloc[-1].close
            
            # open position valuation
            self._data.loc[g.index, OPENPOSITION_VALUATION] = last_day_close_price * g[OPEN_POSITION].iloc[0]
            
            # open trade valuation
            self._data.loc[g.index, OPENTRADE_VALUATION] = abs(g[IS_OPEN_TRADE] * g[TRADE_PRICE] * g[OPENTRADE_SIZE]).cumsum().values
        
        self._data[NET_PNL] = self._data[OPEN_PNL] + self._data[TALLY_PNL] + self._data[COST_PNL]
        
        # compute percentage pnl
        self._data[OPEN_PNL_PERCT] = self._data.apply(lambda x: self.__calculate_pnl_open_perct(x), axis = 1)
        self._data[TALLY_PNL_PERCT] = self._data.apply(lambda x: self.__calculate_pnl_tally_perct(x), axis = 1)
        self._data[NET_PNL_PERCT] = self._data.apply(lambda x: self.__calculate_pnl_net_perct(x), axis = 1)

    
    def run(self):
        self.initialize_pnl_position_columns()
        self.add_transaction_costs()
        self.run_position_trace()
        self.run_pnl_trace()
        return self._data.copy()
    
    @property
    def complete_intraday_position_pnl_book(self):
        return self._data.copy()


