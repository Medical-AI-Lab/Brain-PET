import pandas as pd
import sys
import os
import glob
import re
import subprocess
from tqdm import tqdm


class ROI_maker():
    def __init__(self, args, MPRAGEpath, l_id):
        self.mpragepath = MPRAGEpath
        self.betweenpath = self.mpragepath.split('/',1)[-1].rsplit('/',1)[0]
        self.Reconallname = self.mpragepath.split('/')[-1].rsplit('.',1)[0] #Estimate reconall filename from MPRAGE
        self.Reconalldir = os.path.join(args.freesurferdir, self.Reconallname)
        self.originalmgz = args.freesurferbasefile
        self.fittedmgz = self.originalmgz.replace('.mgz','_T1.mgz')
        self.fittednifti = self.fittedmgz.replace('.mgz','.nii.gz')
        self.mergedroinifti = args.outputfile
        self.freesurfermgzpath = os.path.join(self.Reconalldir, 'mri', self.originalmgz)
        self.roisavedir = os.path.join(args.roidir, args.outputdir, self.betweenpath)
        self.fittedmgzpath = os.path.join(self.roisavedir, self.fittedmgz)
        self.fittedniftipath = os.path.join(self.roisavedir, self.fittednifti)
        self.mergedniftipath = os.path.join(self.roisavedir, self.mergedroinifti)
        self.l_id = l_id

    def makedir(self):
        os.makedirs(self.roisavedir, exist_ok=True)
        
    def fit2mprage(self):
        subprocess.call(["mri_label2vol", "--seg", self.freesurfermgzpath, "--temp", self.mpragepath, "--o", self.fittedmgzpath, "--regheader", self.freesurfermgzpath])

    def mgz2nifti(self):
        subprocess.call(["mrconvert", self.fittedmgzpath, self.fittedniftipath])

    def roi_merger(self):
        l_step = ['step((a-{})*({}-a))'.format(i-0.5, i+0.5) for i in self.l_id]
        steps = "+".join(l_step)
        cmd = "3dcalc -a '{}' -expr '{}' -prefix '{}'".format(self.fittedniftipath, steps, self.mergedniftipath)
        subprocess.call(cmd, shell=True)
    
    def corresponding_series(self):
        l_series = [self.mpragepath, self.freesurfermgzpath, self.mergedniftipath]
        l_index = ['MPRAGEpath','Freesurferpath', 'ROIpath']
        s_corr = pd.Series(l_series, index=l_index)
        return s_corr

def _argparse():
    import argparse
    parser = argparse.ArgumentParser(description='Choose csv and define lists to analyze')
    parser.add_argument('--roicsvpath', type=str, default='./target_roi.csv', help='path to ROICSV')
    parser.add_argument('--freesurfertablepath', type=str, default='./FreeSurferTable.csv', help='path to Freesurfer table csv')
    parser.add_argument('--mpragedir', type=str, default='MPRAGE', help='path to MPRAGE dir')
    parser.add_argument('--freesurferdir', type=str, default='MPRAGE_freesurfer', help='path to Freesurfer dir')
    parser.add_argument('--freesurferbasefile', type=str, default='aparc+aseg.mgz', help='Freesurfer basefile')
    parser.add_argument('--outputfile', type=str, default='merged_roi.mgz', help='Output filename')
    parser.add_argument('--roidir', type=str, default='ROI', help='path to ROI dir')
    parser.add_argument('--outputdir', type=str, default='target', help='What kinds of ROI are you making?')
    parser.add_argument('--filetype', choices=['.nii', '.nii.gz'], help='.nii or .nii.gz?')
    args = parser.parse_args()
    return args

def MPRAGE_freesurfer_check(MPRAGEdir, freesurferdir, filetype):
    l_MPRAGEpath = [p for p in glob.glob(MPRAGEdir + '/**', recursive=True) if re.search(filetype, p)]
    l_MPRAGEname = [p.split('/')[-1] for p in l_MPRAGEpath]

    dirfromMPRAGE = [f.rsplit('.',1)[0] for f in l_MPRAGEname]
    dirReconAll = os.listdir(freesurferdir)
    bothdirnumber = pd.Series(dirReconAll).isin(pd.Series(dirfromMPRAGE)).sum()
    l_diff = set(dirfromMPRAGE) ^ set(dirReconAll)

    print("If following differences are accecptable, any keys. ")
    print(l_diff)
    input()
    return l_MPRAGEpath #MPRAGE-basis is better than both-basis.

def Label2ID(sourcetable, regionlist):
    df_src = pd.read_csv(sourcetable)
    df_list = pd.read_csv(regionlist)
    df_merge = pd.merge(df_src, df_list, on='LabelName')
    if len(df_merge) == len(df_list):
        l_id = list(df_merge['No'])
    else:
        sys.exit(1)
    return l_id

def main(args):
    l_MPRAGEpath = MPRAGE_freesurfer_check(args.mpragedir, args.freesurferdir, args.filetype)
    l_id = Label2ID(args.freesurfertablepath, args.roicsvpath)
    df = pd.DataFrame()
    for MPRAGEpath in tqdm(l_MPRAGEpath): #MPRAGE-basis
        roi_maker = ROI_maker(args, MPRAGEpath, l_id)
        roi_maker.makedir()
        roi_maker.fit2mprage()
        roi_maker.mgz2nifti()
        roi_maker.roi_merger()
        corresponding_series = roi_maker.corresponding_series()
        df = pd.concat([df, corresponding_series],axis=1)
    return df.T

if __name__ == '__main__':
    args =_argparse()
    df_corr = main(args)
    corr_csvpath  = os.path.join(args.roidir, args.outputdir + '_corr.csv')
    df_corr.to_csv(corr_csvpath, index=False)
