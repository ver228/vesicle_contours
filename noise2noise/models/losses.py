#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 10:27:31 2018

@author: avelinojaver
"""

from torch import nn

class AnnelingLoss(nn.Module):
    def __init__(self, anneling_rate=1/50):
        super().__init__()
        
        self.anneling_rate = anneling_rate
        self._n_calls = 0
        self._init_gamma = 2
        self._last_gamma = 0
        self._eps = 1e-8
    
    def forward(self, input, target):
        gamma = max(self._init_gamma - self._n_class*self.anneling_rate, self._last_gamma)
        self._n_calls += 1
        
        return ((input-target).abs() + self._eps).pow(gamma).sum()

