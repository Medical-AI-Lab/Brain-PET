import os
import glob
from scipy import stats
import pandas as pd
import numpy as np
from tqdm import tqdm


class Correlation():
    def __init__(self, df):
        df_mean = df.dropna(subset=['fake_tumor_mean_ratio'])
        self.real_mean = df_mean['original_tumor_mean_ratio']
        self.fake_mean = df_mean['fake_tumor_mean_ratio']
        df_max = df.dropna(subset=['fake_tumor_max_ratio'])
        self.real_max = df_max['original_tumor_max_ratio']
        self.fake_max = df_max['fake_tumor_max_ratio']
        df_vol = df.dropna(subset=['fake_tumor_volume'])
        self.real_vol = df_vol['original_tumor_volume']
        self.fake_vol = df_vol['fake_tumor_volume']

    def pearsonr_with_ci(x, y, alpha=0.05):
        r, p = stats.pearsonr(x, y)
        r_z = np.arctanh(r)
        se = 1 / np.sqrt(x.size - 3)
        z = stats.norm.ppf(1 - alpha/2)
        lo_z, hi_z = r_z - z*se, r_z + z*se
        lo, hi = np.tanh((lo_z, hi_z))
        return r, p, lo, hi

    def spearman_with_ci(x, y, alpha=0.05):
        r, p = stats.spearmanr(x, y)
        r_z = np.arctanh(r)
        se = 1 / np.sqrt(x.size - 3)
        z = stats.norm.ppf(1 - alpha/2)
        lo_z, hi_z = r_z - z*se, r_z + z*se
        lo, hi = np.tanh((lo_z, hi_z))
        return r, p, lo, hi

    def corr_set_calc(self):
        if (self.fake_mean.notna() & self.real_mean.notna()).any():
            p_mean_score, p_mean_pvalue, p_mean_lower, p_mean_upper = Correlation.pearsonr_with_ci(
                self.real_mean, self.fake_mean)
            s_mean_score, s_mean_pvalue, s_mean_lower, s_mean_upper = Correlation.spearman_with_ci(
                self.real_mean, self.fake_mean)
        else:  # 全部がnanの可能性があり、そのときにはpd.NAにする。
            p_mean_score = p_mean_pvalue = p_mean_lower = p_mean_upper = s_mean_score = s_mean_pvalue = s_mean_lower = s_mean_upper = pd.NA

        if (self.fake_max.notna() & self.real_max.notna()).any():
            p_max_score, p_max_pvalue, p_max_lower, p_max_upper = Correlation.pearsonr_with_ci(
                self.real_max, self.fake_max)
            s_max_score, s_max_pvalue, s_max_lower, s_max_upper = Correlation.spearman_with_ci(
                self.real_max, self.fake_max)
        else:
            p_max_score = p_max_pvalue = p_max_lower = p_max_upper = s_max_score = s_max_pvalue = s_max_lower = s_max_upper = pd.NA

        if (self.real_vol.notna() & self.fake_vol.notna()).any():
            p_vol_score, p_vol_pvalue, p_vol_lower, p_vol_upper = Correlation.pearsonr_with_ci(
                self.real_vol, self.fake_vol)
            s_vol_score, s_vol_pvalue, s_vol_lower, s_vol_upper = Correlation.spearman_with_ci(
                self.real_vol, self.fake_vol)
        else:
            p_vol_score = p_vol_pvalue = p_vol_lower = p_vol_upper = s_vol_score = s_vol_pvalue = s_vol_lower = s_vol_upper = pd.NA

        s_corr = pd.Series([p_mean_score, p_mean_pvalue, p_mean_lower, p_mean_upper,
                            p_max_score, p_max_pvalue, p_max_lower, p_max_upper,
                            p_vol_score, p_vol_pvalue, p_vol_lower, p_vol_upper,
                            s_mean_score, s_mean_pvalue, s_mean_lower, s_mean_upper,
                            s_max_score, s_max_pvalue, s_max_lower, s_max_upper,
                            s_vol_score, s_vol_pvalue, s_vol_lower, s_vol_upper
                            ], index=l_target)
        return s_corr


def main(realcsv, nifti_dir, fake_dir, fakecsv):
    df_real = pd.read_csv(realcsv)
    pattern_dir = os.path.join(nifti_dir, fake_dir)

    l_target = ['pearson_corcoe_mean_score', 'pearson_corcoe_mean_pvalue', 'pearson_corcoe_mean_lower', 'pearson_corcoe_mean_upper',
                'pearson_corcoe_max_score', 'pearson_corcoe_max_pvalue', 'pearson_corcoe_max_lower', 'pearson_corcoe_max_upper',
                'pearson_corcoe_vol_score', 'pearson_corcoe_vol_pvalue', 'pearson_corcoe_vol_lower', 'pearson_corcoe_vol_upper',
                'spearman_corcoe_mean_score', 'spearman_corcoe_mean_pvalue', 'spearman_corcoe_mean_lower', 'spearman_corcoe_mean_upper',
                'spearman_corcoe_max_score', 'spearman_corcoe_max_pvalue', 'spearman_corcoe_max_lower', 'spearman_corcoe_maxa_upper',
                'spearman_corcoe_vol_score', 'spearman_corcoe_vol_pvalue', 'spearman_corcoe_vol_lower', 'spearman_corcoe_vol_upper']

    for pattern in glob.glob(pattern_dir + '/*'):
        for split in glob.glob(pattern + '/test'):
            df_summary = pd.DataFrame(columns=l_target)
            for epoch in tqdm(glob.glob(split + '/*')):
                target_dir = os.path.join(epoch, '8bit')
                fakepath = os.path.join(target_dir, fakecsv)
                df_fake = pd.read_csv(fakepath)
                df_merge = pd.merge(
                    df_real, df_fake, on='Filename', how='left')
                calc_correlation = Correlation(df_merge)
                _s_tmp = calc_correlation.corr_set_calc()
                _s_tmp.name = epoch
                df_summary = df_summary.append(_s_tmp)
    df_summary.to_csv('summary.csv')

if __name__ == '__main__':
    realcsv = 'real_results.csv'
    nifti_dir = './nifti'
    fake_dir = 'fake'
    fakecsv = 'fake_results.csv'
    main(realcsv, nifti_dir, fake_dir, fakecsv)
