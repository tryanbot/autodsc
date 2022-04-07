from typing import Optional
import numpy as np

class Interpolate:
  def __init__(
    self, x_list, 
    y_list, 
    left: Optional[float] = None, 
    right: Optional[float] = None, 
    period: Optional[float] = None
    ):

    self.x_list = x_list
    self.y_list = y_list
    self.left = left
    self.right = right
    self.period = period
  
  def __call__(self, x):
    return np.interp(x, self.x_list, self.y_list, self.left, self.right, self.period)