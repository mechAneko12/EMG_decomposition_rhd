import pandas as pd
import numpy as np
from emgineer import EmgDecomposition, plot_spikes
import matplotlib.pyplot as plt
import os

class EmgDataset():
    def __init__(self, data_path, l_emg=None) -> None:
        self.df = self._load_data_file(data_path, ['emg_' + str(x) for x in range(128)])
        self.emg_flex_raw = self.df[['emg_' + str(x) for x in range(64)]]
        self.emg_ext_raw = self.df[['emg_' + str(x) for x in range(64, 128)]]
        if l_emg is not None:
            self.emg_flex_raw = self.emg_flex_raw[:l_emg]
            self.emg_ext_raw = self.emg_ext_raw[:l_emg]
        
    def _load_data_file(self, filepath, names):
        df = pd.read_csv(filepath,
                        header=None,
                        names=names)
        return df
    
    
class emg2spike(EmgDataset):
    def __init__(self, data_path, n_motor_unit, l_emg=None, cashe_name='sample') -> None:
        super().__init__(data_path, l_emg=l_emg)
        self.n_motor_unit = n_motor_unit
        self.cashe_name = cashe_name
        
    def main(self, flag_sil=True):
        est_flex = EmgDecomposition(n_motor_unit=self.n_motor_unit,
                               random_state=0,
                               cashe=self.cashe_name + '_flex',
                               flag_sil=flag_sil,
                               flag_pca=True)
        est_flex.fit(self.emg_flex_raw)
        st_flex, _ = est_flex.transform(self.emg_flex_raw)
        
        est_ext = EmgDecomposition(n_motor_unit=self.n_motor_unit,
                               random_state=0,
                               cashe=self.cashe_name + '_ext',
                               flag_sil=flag_sil,
                               flag_pca=True)
        est_ext.fit(self.emg_ext_raw)
        st_ext, _ = est_ext.transform(self.emg_ext_raw)
        return st_flex, st_ext
    
    
if __name__ == '__main__':
    n = 2
    file = 'exp0712-' + str(n)
    #file = 'exp0712_mix-2-3'
    filepath = 'dataset/20220712/' + file + '.csv'
    e2s = emg2spike(filepath, 50, cashe_name=file)
    
    st_flex, st_ext = e2s.main(flag_sil=True)
    df_flex = pd.DataFrame(st_flex); df_ext = pd.DataFrame(st_ext)
    #df_flex.to_csv('output/20220712/st_flex_exp0712-' + str(n) + '.csv')
    #df_ext.to_csv('output/20220712/st_ext_exp0712-' + str(n) + '.csv')
    
    #df_flex.to_csv('output/20220712/st_flex_exp0712-2-3.csv')
    #df_flex.to_csv('output/20220712/st_ext_exp0712-2-3.csv')
    
    plot_spikes(st_flex, title='st_flex_expdata' + str(n))
    plot_spikes(st_ext, title='st_ext_expdata' + str(n))
    