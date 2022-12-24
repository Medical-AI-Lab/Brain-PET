import pandas as pd
import os
import glob
import re
import subprocess
from tqdm import tqdm

def _argparse():
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess for MPRAGEs')
    parser.add_argument('--mpragedir', type=str, default='MPRAGE', help='path to MPRAGE dir')
    parser.add_argument('--maindir', type=str, default='MPRAGE_processed', help='Main directory')
    parser.add_argument('--resampleddir', type=str, default='resampled', help='Resampled sub-directory')
    parser.add_argument('--harmonizeddir', type=str, default='harmonized', help='Harmonized sub-directory')
    parser.add_argument('--brainextracteddir', type=str, default='brainextracted', help='Brainextracted sub-directory')
    parser.add_argument('--normalizeddir', type=str, default='normalized', help='Normalized sub-directory')
    parser.add_argument('--resamplesize', type=int, default=1, help='Resample size')
    parser.add_argument('--gpudevice', type=bool, default=True, help='GPU?')

    parser.add_argument('--filetype', choices=['.nii', '.nii.gz'], help='.nii or .nii.gz?')
    args = parser.parse_args()
    return args


class MRI_preprocess():
    def __init__(self, args, MPRAGEpath):
        self.mpragepath = MPRAGEpath
        if args.filetype == '.nii':
            self.mpragepath = self.mpragepath.replace('.nii','.nii.gz')
        self.mpragefile = self.mpragepath.rsplit('/')[-1]
        self.maindir = args.maindir
        self.resampledfilepath = self.mpragepath.replace(args.mpragedir, self.maindir+'/'+args.resampleddir)
        self.harmonizedpath = self.mpragepath.replace(args.mpragedir, self.maindir+'/'+args.harmonizeddir)
        self.brainextractedpath = self.mpragepath.replace(args.mpragedir, self.maindir+'/'+args.brainextracteddir)
        self.brainextractedmaskpath = self.mpragepath.replace(args.mpragedir, self.maindir+'/'+args.brainextracteddir).replace(args.filetype,'_mask' + args.filetype)
        self.normalizedpath = self.mpragepath.replace(args.mpragedir, self.maindir+'/'+args.normalizeddir)
        self.resamplesize = args.resamplesize
        self.gpudevice = args.gpudevice

        self.resampledfiledir = self.mpragepath.replace(args.mpragedir, self.maindir+'/'+args.resampleddir).rsplit('/',1)[0]
        self.harmonizeddir = self.mpragepath.replace(args.mpragedir, self.maindir+'/'+args.harmonizeddir).rsplit('/',1)[0]
        self.brainextracteddir = self.mpragepath.replace(args.mpragedir, self.maindir+'/'+args.brainextracteddir).rsplit('/',1)[0]
        self.normalizeddir = self.mpragepath.replace(args.mpragedir, self.maindir+'/'+args.normalizeddir).rsplit('/',1)[0]
        os.makedirs(self.resampledfiledir, exist_ok=True)
        os.makedirs(self.harmonizeddir, exist_ok=True)
        os.makedirs(self.brainextracteddir, exist_ok=True)
        os.makedirs(self.normalizeddir, exist_ok=True)

    def resampling(self):
        cmd = "flirt -in '{}' -ref '{}' -out '{}' -applyisoxfm '{}'".format(self.mpragepath, self.mpragepath, self.resampledfilepath, self.resamplesize)
        subprocess.call(cmd, shell=True)

    def harmonization(self):
        cmd = "N4BiasFieldCorrection -i '{}' -o '{}' -d 3 -v 1 -s 4 -b [ 180 ] -c [ 50x50x50x50, 0.0 ]".format(self.resampledfilepath, self.harmonizedpath)
        subprocess.call(cmd, shell=True)

    def brainextract(self):
        if self.gpudevice:
            cmd = "hd-bet -i '{}' -o '{}'".format(self.harmonizedpath, self.brainextractedpath)
        else:
            cmd = "hd-bet -i '{}' -o '{}' -device cpu -mode fast -tta 0".format(self.harmonizedpath, self.brainextractedpath)
        subprocess.call(cmd, shell=True)
        
    def normalization(self):
        cmd = "fcm-normalize '{}' -o '{}' -m '{}' -mo t1 -tt wm -v".format(self.brainextractedpath, self.normalizedpath, self.brainextractedmaskpath)
        subprocess.call(cmd, shell=True)

def main(args):
    l_MPRAGEpath = [p for p in glob.glob(args.mpragedir + '/**', recursive=True) if re.search(args.filetype, p)]
    for MPRAGEpath in tqdm(l_MPRAGEpath): #MPRAGE-basis
        processor = MRI_preprocess(args, MPRAGEpath)
        processor.resampling()
        processor.harmonization()
        processor.brainextract()
        processor.normalization()


if __name__ == '__main__':
    args =_argparse()
    main(args)
