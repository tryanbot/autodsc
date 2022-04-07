from typing import Optional
import pandas as pd
import numpy as np


class PseudoLabeler:
  def __init__(
    self,
    df_train,
    target: str,
    sample_rate: Optional[float] = 0.5,
    confidence: Optional[float] = 0.5 
  ):
    print('halo')
    self.df_train = df_train
    self.target = target
    self.sample_rate = sample_rate
    self.confidence = confidence


  def tabular(self, pseudo_label, X_test):
    print('halo')
    pseudo_data = X_test.copy(deep=True)
    pseudo_data[self.target] = pseudo_label
    sampled_pseudo_data = pseudo_data.sample(frac=self.sample_rate)
    print(sampled_pseudo_data.reset_index(drop=True))
    print(self.df_train.reset_index(drop=True))
    
    augmented_train = pd.concat([sampled_pseudo_data.reset_index(drop=True), self.df_train.reset_index(drop=True)], ignore_index=True)

    return augmented_train
    

  def image():
    pass

  def video():
    pass

  def text():
    pass