import datetime
from collections import deque

class Entry(object):
    """
    Defines an accounting entry.
    """
    def __init__(self, quantity, price, factor=1, **kwargs):
        """
        Initializes an entry object with quantity, price and arbitrary
        data associated with the entry to be used post-fifo-accounting
        analysis purposes.
        Note the factor parameter. This prameter is applied to the
        price.
        
        factor is not really useful at current stage
        
        TODO:
        1. add buy/sell's time (in order to compute duration of matched trade in fifo)
        
        """
        
        
        ## Save data slots:
        self.quantity = quantity
        self.price = price
        self.factor = factor
        self.data = kwargs

    def __repr__(self):
        return "%s @%s" % (self.quantity, self.price)

    @property
    def size(self):
        return abs(self.quantity)

    @property
    def trade_size(self):
        return self.quantity

    @property
    def trade_price(self):
        return self.price

    @property
    def buy(self):
        return self.quantity > 0

    @property
    def sell(self):
        return not self.buy

    @property
    def zero(self):
        return self.quantity == 0

    @property
    def value(self):
        return self.quantity * self.price * self.factor

    def copy(self, quantity=None):
        return Entry(quantity or self.quantity, self.price, self.factor, **self.data.copy())


class FIFO(object):
    """
    Implements a FIFO accounting rule by (1) calculating the cost of
    the inventory in hand, (2) calculating the historical PnL trace.
    
    TODO:
        1. compute holding time (duration between buy and sell)
    """

    def __init__(self, entries=None):
        """
        Initializes and computes the FIFO accounting.
        Note that entries are supposed to be sorted.
        """
        ## Mark the start timestamp:
        self._started_at = datetime.datetime.now()
        self._finished_at = None

        ## Save data slots:
        self._entries = entries or []

        ## Declare and initialize private fields to be used during computing:
        self._balance = 0
        self.inventory = deque()
        self.trace = []

        ## Start computing:
        self._compute()

    @property
    def is_empty(self):
        """
        Indicates if the inventory is empty.
        """
        return len(self.inventory) == 0

    @property
    def stock(self):
        """
        Returns the available stock.
        """
        return self._balance

    @property
    def valuation(self):
        """
        Returns the inventory valuation.
        """
        return sum([s.quantity * s.price for s in self.inventory])

    @property
    def valuation_factored(self):
        """
        Returns the inventory valuation which is factored.
        """
        return sum([s.quantity * s.price * s.factor for s in self.inventory])

    @property
    def profit_and_loss(self):
        """
        Returns the realized profit and loss.
        """
        return sum([-e.price * e.quantity for entries_lst in self.trace for e in entries_lst])

    @property
    def profit_and_loss_factored(self):
        """
        Returns the realized profit and loss.
        """
        return - sum(
            [-e.price * e.quantity * e.factor
             for entries_lst in self.trace
             for e in entries_lst
             ]
        )

    @property
    def winning_rate(self):
        # whether a trade is profit
        trade_is_profit = lambda buy_price, sell_price: 1 if (sell_price > buy_price) else 0
        num_of_profit_trade = sum(
            list(map(lambda e: trade_is_profit(e[0].trade_price, e[1].trade_price), self.trace))
        )
        return round(num_of_profit_trade / len(self.trace), 3)

    @property
    def avgcost(self):
        """
        Returns the average cost of the inventory.
        """
        ## If we don't have any stock, simply return None, else average:
        return None if self._balance == 0 else (self.valuation / self._balance)

    @property
    def avgcost_factored(self):
        """
        Returns the average cost of the inventory which is factored.
        """
        ## If we don't have any stock, simply return None, else average:
        return None if self._balance == 0 else (self.valuation_factored / self._balance)

    @property
    def runtime(self):
        """
        Returns the total runtime.
        """
        if self._started_at is not None and self._finished_at is not None:
            return self._finished_at - self._started_at
        return None

    def _push(self, entry):
        """
        Pushes the entry to the inventory as new stock movement.
        """
        self.inventory.append(entry)
        self._balance += entry.quantity

    def _fill(self, entry):
        """
        Fills existing stock entries by calculating new stocks if required.
        """
        ## OK, we know that this is a contra-entry for our existing
        ## stock entries, ie. if our balance is positive, this is
        ## negative, or vice-versa. Keep in mind that it may even be
        ## bigger in quantity compared to out balance which will
        ## eventually reverse the sign of our balance, like selling
        ## 100 items when we have stock only for 50. This function
        ## will deal with these situations and calculate new stock
        ## entries.
        ##
        ## OK, let's start with this munch-fill-reverse cycle by
        ## creating a copy of the entry:
        entry = entry.copy()

        ## We will continue as long as the entry has quantity:
        while not entry.zero:
            ## Let's consume the earliest entry from the
            ## inventory. But, if the inventory is empty, we can then
            ## safely push the entry to the inventory:
            if self.is_empty:
                ## Yes, the inventory is empty. Push:
                self._push(entry)

                ## We are done here now! Return:
                return

            ## We have entries in the inventory. Get the earliest:
            earliest = self.inventory.popleft()

            ## There are 3 possible cases:
            ##
            ## 1. entry.size < earliest.size  : Munch from earliest, put earliest back and return
            ## 2. entry.size == earliest.size : Remove the earliest entirely and return
            ## 3. entry.size > earliest.size  : Remove the earliest, adjust entry and continue cycle
            ##
            ## Note that in any of these cases we will update the
            ## trace, too. Let's start:
            if entry.size <= earliest.size:
                ## We will now munch from the earliest:
                munched = earliest.copy(-entry.quantity)

                ## Update the earliest:
                earliest.quantity += entry.quantity

                ## Put earliest back to the inventory if still have quantity:
                if earliest.quantity != 0:
                    self.inventory.appendleft(earliest)

                ## Update the trace:
                self.trace.append([munched, entry])

                ## Update the balance:
                self._balance += entry.quantity

                ## Done, return:
                return
            else:
                ## Munch from the entry:
                munched = entry.copy(-earliest.quantity)

                ## Update the entry:
                entry.quantity += earliest.quantity

                ## Update the trace:
                self.trace.append([earliest, munched])

                ## Update the balance and continue:
                self._balance += munched.quantity

    def _compute(self):
        """
        Computes the FIFO accounting for the given entries and produces
        the (1) cost of the inventory in hand, (2) historical PnL trace.
        """
        ## We will iterate over the entries and operate on the
        ## inventory. Let's start:
        for entry in self._entries:
            ## We will add new stock to the inventory or remove
            ## existing stock from the inventory. It looks pretty
            ## straight-forward. But is it?
            ##
            ## There is a special case which is called "short-selling"
            ## in the financial jargon. This is similar to backorders
            ## in the convential trading of goods which means selling
            ## goods which you don't have in your inventory yet.
            ##
            ## This means that we have the following possible
            ## situations:
            ##
            ## | Stock    | Entry |
            ## |----------|-------|
            ## | positive | buy   |
            ## | positive | sell  |
            ## | negative | buy   |
            ## | negative | sell  |
            ##
            ## As you see, there are two cases which are pretty easy
            ## to handle:
            ##
            ## | Stock    | Entry | Action        |
            ## |----------|-------|---------------|
            ## | positive | buy   | Keep adding   |
            ## | negative | sell  | Keep removing |
            ##
            ## Let's do this:
            if (self._balance >= 0 and entry.buy) or (self._balance <= 0 and entry.sell):
                ## Yes, we will push the entry to the inventory as is:
                self._push(entry)
            ## Good, we will now proceed with the more complicated
            ## operation: Closing previously opened stock
            ## positions. This applies to the following cases with the
            ## required actions to be taken respectively.
            ##
            ## | Stock    | Entry | Action                                     |
            ## |----------|-------|--------------------------------------------|
            ## | positive | sell  | Munch from stock (and reverse if required) |
            ## | negative | buy   | Fill backorders (and reverse if required)  |
            ##
            ## Note that we must make sure that we skip "0"-quantity entries.
            elif not entry.zero:
                ## OK, the entry is not zero. We will proceeding
                ## filling positions:
                self._fill(entry)

            ## We are done with the entry. Let's move to the next one.

        ## This marks the end of the the FIFO computation:
        self._finished_at = datetime.datetime.now()


if __name__ == "__main__":
    # Create FIFO queue entry for each trade
    #   +75 MSFT @25.10
    #   +50 MSFT @25.12
    #  -100 MSFT @25.22

    fifo = FIFO(
        [Entry(+75, 25.10), Entry(+50, 25.12),
                 Entry(0, 25.12), Entry(0, 100), # zero quantity
                 Entry(-100, 25.22), Entry(-20, 25.02), Entry(-20, 25.01)]
    )

    print("Available Stock          : ", fifo.stock)
    print("Stock Valuation          : ", fifo.valuation)
    print("Average Cost    : ", fifo.avgcost)
    print("FIFO (matched) trades : ", fifo.trace)
    print("Realized PnL is : ", fifo.profit_and_loss)
    print("Winning rate is : ", fifo.winning_rate)
    trans_lookup_idx = 2
    print(f"The transaction #{trans_lookup_idx} has quantity = {fifo.trace[trans_lookup_idx - 1][0].size} "
          f"with buying at {fifo.trace[trans_lookup_idx - 1][0].trade_price} and selling at {fifo.trace[trans_lookup_idx - 1][1].trade_price}")

    #print("Trace Length             : ", len(fifo.trace))
    #print("Total Runtime            : ", fifo.runtime)
    # print("Factored Stock Valuation : ", fifo.valuation_factored)
    # print("Factored Average Cost             : ", fifo.avgcost_factored)



