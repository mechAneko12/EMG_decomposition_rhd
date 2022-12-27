import pandas as pd
import numpy as np

def st2fr(df_st: pd.DataFrame, len_sample: int=768, is_non_overlap=True) -> pd.DataFrame:
    """spike trsins to firing rate.

    Args:
        df_st (pd.DataFrame): DataFrame of spike trains.
        len_sample (int, optional): length of each sample. Defaults to 768.
        is_non_overlap (bool, optional): Each sample overlaps or not. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame of firing rate.
    """
    if is_non_overlap:
        df_fr = df_st.groupby(df_st.index // len_sample).sum()
    else:
        df_fr = df_fr.rolling(len_sample).sum().dropna()
    return df_fr

def label_fr(df_fr: pd.DataFrame, n_repeat: int, min_fr: int=10) -> pd.DataFrame:
    df_fr['label'] = df_fr.sum(axis=1)
    df_fr['label'] = df_fr['label'].map(lambda x: 1 if x > min_fr else 0)

    index_transition = df_fr[df_fr['label'].diff().fillna(0) != 0].index.tolist()
    assert len(index_transition) != 2 * n_repeat

    list_label = [_label(i_row, index_transition, n_repeat) for i_row in range(len(df_fr))]
    df_fr['label'] = list_label
    
    return df_fr

def _label(i_row, index_transition, n_repeat):
    for j, _index_transition in enumerate(index_transition):
        # index_transitionのどこの区間の間か
        if i_row < _index_transition:
            break
    
    # 偶数の区間では0を返す
    if j % 2 == 0:
        return 0
    # 動作の開始・終了:2, 同じ動作の繰り返し回数:n_repeatをかけた数で割って1を足して動作のindex
    else:
        return j // (2 * n_repeat) + 1
    


if __name__ == '__main__':
    df_st = pd.read_csv('')[2000:]
    df_fr = st2fr(df_st)
    df_fr_labeled = label_fr(df_fr, n_repeat=3, min_fr=10)

