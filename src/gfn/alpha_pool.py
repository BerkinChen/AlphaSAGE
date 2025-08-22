from typing import List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor
from alphagen.models.alpha_pool import AlphaPool, AlphaPoolBase
from alphagen.data.expression import Expression
from alphagen_qlib.stock_data import StockData


class AlphaPoolGFN(AlphaPool):
    def __init__(
        self,
        capacity: int,
        stock_data: StockData,
        target: Expression,
        ic_mut_threshold: float = 0.3
    ):
        super().__init__(capacity, stock_data, target)
        self.ic_mut_threshold = ic_mut_threshold

    def try_new_expr(self, expr: Expression) -> float:
        value = self._normalize_by_day(expr.evaluate(self.data))
        ic_ret, ic_mut = self._calc_ics(value, ic_mut_threshold=0.99)
        if ic_ret is None or ic_mut is None:
            return -1.
        
        # Check if we should add this factor to the pool
        if self.size < self.capacity:
            # Pool not full, add directly if correlation constraint is satisfied
            if not ic_mut or max(ic_mut) <= self.ic_mut_threshold:
                self._add_factor(expr, value, ic_ret, ic_mut)
        else:
            # Pool is full, check if this factor is better than the worst one
            min_ic_idx = np.argmin(self.single_ics[:self.size])
            min_ic = self.single_ics[min_ic_idx]
            
            if ic_ret > min_ic and (not ic_mut or max(ic_mut) <= self.ic_mut_threshold):
                # Remove the worst factor
                self._pop()
                # Add the new factor
                self._add_factor(expr, value, ic_ret, ic_mut)
        
        return ic_ret
    
    def _pop(self) -> None:
        # Pop the factor with the lowest ic
        if self.size <= self.capacity:
            return
        idx = np.argmin(self.single_ics[:self.size])
        self._swap_idx(idx, self.capacity)
        self.size = self.capacity
