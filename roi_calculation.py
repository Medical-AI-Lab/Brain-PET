import os
import glob
import subprocess
import pandas as pd
from tqdm import tqdm


class Threthold():
    def __init__(self, s, target_dir, ref_dir):
        self.name = s['Filename']
        self.pre = s['pre']
        self.ref = self.pre.replace('pre', 'ref')
        self.targetpath = os.path.join(target_dir, self.name)
        self.refpath = os.path.join(ref_dir, self.ref)

    def normal_meanvalue(self):
        mean_value_str = subprocess.check_output(
            ["3dmaskave", "-quiet", "-mask", self.refpath, self.targetpath]).decode().replace('\n', '')
        mean_value_float = float(mean_value_str)
        return mean_value_float


class Calculation():
    def __init__(self, s, target_dir, ref_dir):
        self.name = s['Filename']
        self.pre = s['pre']
        self.mask = s['mask']
        self.threshold = s['threshold']
        self.ref = self.pre.replace('pre', 'ref')
        self.targetpath = os.path.join(target_dir, self.name)
        self.refpath = os.path.join(ref_dir, self.ref)
        self.fake_mask = self.mask.replace('_mask', '_fake_mask')
        self.fake_mask_path = os.path.join(target_dir, self.fake_mask)
        self.pre_path = os.path.join(ref_dir, self.pre)

    def makefakemask(self):
        # self.thresholdをそのまま下のstepのとこに代入するとエラーになる。
        threshold = self.threshold
        subprocess.call(["3dcalc", "-a", self.targetpath, "-b", self.pre_path,
                        "-expr", "step((a - threshold)*b)", "-prefix", self.fake_mask_path])

    def suv_mean_roi(self):
        if os.path.isfile(self.fake_mask_path):
            try:
                # ./nifti/fake/test_resize_crop_512x256_8bit_3ch_1ch_cropped-whole-4-6_epoch-200_2022-06-10-18-15-31/test/test_100で出るエラーへの対処。collapseでmaskが変なんかも。
                mean_roi_str = subprocess.check_output(
                    ["3dmaskave", "-quiet", "-mask", self.fake_mask_path, self.targetpath]).decode().replace('\n', '')
                mean_roi_float = float(mean_roi_str)
            except subprocess.CalledProcessError as e:
                mean_roi_float = pd.NA
                print(e)
        else:
            mean_roi_float = pd.NA
        return mean_roi_float

    def suv_max_roi(self):
        if os.path.isfile(self.fake_mask_path):
            try:
                max_roi_str = subprocess.check_output(
                    ["3dmaskave", "-quiet", "-max", "-mask", self.fake_mask_path, self.targetpath]).decode().replace('\n', '')
                max_roi_float = float(max_roi_str)
            except subprocess.CalledProcessError as e:
                max_roi_float = pd.NA
                print(e)
        else:
            max_roi_float = pd.NA
        return max_roi_float

    def suv_vol_roi(self):
        if os.path.isfile(self.fake_mask_path):
            try:
                vol_roi_str = subprocess.check_output(
                    ["3dBrickStat", "-volume", "-mask", self.fake_mask_path, self.fake_mask_path]).decode().replace('\n', '')
                vol_roi_float = float(vol_roi_str)
            except subprocess.CalledProcessError as e:
                vol_roi_float = pd.NA
                print(e)
        else:
            vol_roi_float = pd.NA
        return vol_roi_float

def main(nifti_dir, fake_dir, corrcsv):
    df_cor = pd.read_csv(corrcsv)

    pattern_dir = os.path.join(nifti_dir, fake_dir)
    ref_dir = os.path.join(nifti_dir, 'mask/roi_all')

    for pattern in glob.glob(pattern_dir + '/*'):
        for split in glob.glob(pattern + '/test'):
            for epoch in tqdm(glob.glob(split + '/*')):
                target_dir = os.path.join(epoch, '8bit/linear_dsize')

                # threshold
                l_target = ['Filename', 'fake_normal_meanvalue']
                _df_tmp = pd.DataFrame(columns=l_target)
                for i, row in df_cor.iterrows():
                    name = row['Filename']
                    fake_threthold = Threthold(row, target_dir, ref_dir)
                    mean_value_float = fake_threthold.normal_meanvalue()
                    _s_tmp = pd.Series(
                        [name, mean_value_float], index=l_target)
                    _df_tmp = _df_tmp.append(_s_tmp, ignore_index=True)
                df_merge = pd.merge(df_cor, _df_tmp, on='Filename', how='left')
                df_merge['threshold'] = df_merge['fake_normal_meanvalue'] * 1.3
                csvpath = os.path.join(epoch, '8bit', 'threshold.csv')
                df_merge.to_csv(csvpath, index=False)

                # calc
                l_target = ['Filename', 'fake_tumor_mean_ratio',
                            'fake_tumor_max_ratio', 'fake_tumor_volume']
                _df_tmp = pd.DataFrame(columns=l_target)
                for i, row in df_merge.iterrows():
                    name = row['Filename']
                    fake_calculation = Calculation(row, target_dir, ref_dir)
                    fake_calculation.makefakemask()
                    mean_roi_float = fake_calculation.suv_mean_roi()
                    max_roi_float = fake_calculation.suv_max_roi()
                    vol_roi_float = fake_calculation.suv_vol_roi()
                    _s_tmp = pd.Series(
                        [name, mean_roi_float, max_roi_float, vol_roi_float], index=l_target)
                    _df_tmp = _df_tmp.append(_s_tmp, ignore_index=True)

                df_all = pd.merge(df_merge, _df_tmp, on='Filename', how='left')
                csvpath = os.path.join(epoch, '8bit', 'fake_results.csv')
                df_all.to_csv(csvpath, index=False)


if __name__ == '__main__':
    nifti_dir = './nifti'
    fake_dir = 'fake'
    corrcsv = 'pair_table.csv'
    main(nifti_dir, fake_dir, corrcsv)