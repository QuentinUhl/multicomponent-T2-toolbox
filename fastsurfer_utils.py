import os
import itertools as itt
import shutil
import nibabel as nib
import numpy as np
# from IPython import embed


def fastsurfer_volumetric_segmentation(sub_ses_path, t1, FastSurferBin, cpus):
    segmentation = f'{sub_ses_path}/fastsurfer/mri/aparc.DKTatlas+aseg.deep.mgz'
    if not os.path.isfile(segmentation):
        # Switch working directory
        fastsurfer_call = f'{FastSurferBin}/run_fastsurfer.sh --sid fastsurfer --sd {sub_ses_path} --t1 {t1} --seg_only --threads {cpus}'
        current_dir = os.getcwd()
        # Go in FastSurfer directory
        os.chdir(FastSurferBin)
        # Run FastSurfer
        os.system(fastsurfer_call)
        # Go back to original directory
        os.chdir(current_dir)
        # Raise error if segmentation not found
        if not os.path.isfile(segmentation):
            raise FileNotFoundError('~/mri/aparc.DKTatlas+aseg.deep.mgz not found')
        else:
            segmentation = f'{sub_ses_path}/fastsurfer/mri/aparc.DKTatlas+aseg.deep.mgz'
    else:
        print(f'Segmentation found: {segmentation}')
    return segmentation


def fastsurfer_volumetric_surf(sub_ses_path, t1, FastSurferBin, cpus):
    surface_file = f'{sub_ses_path}/fastsurfer/surf/rh.pial'
    if not os.path.isfile(surface_file):
        # Switch working directory
        fastsurfer_call = f'{FastSurferBin}/run_fastsurfer.sh --sid fastsurfer --sd {sub_ses_path} --t1 {t1} --surf_only --threads {cpus}'
        current_dir = os.getcwd()
        # Go in FastSurfer directory
        os.chdir(FastSurferBin)
        # Run FastSurfer
        os.system(fastsurfer_call)
        # Go back to original directory
        os.chdir(current_dir)
        # Raise error if segmentation not found
        if not os.path.isfile(surface_file):
            raise FileNotFoundError('~/fastsurfer/surf/rh.pial not found')
    else:
        print(f'Segmentation found: {surface_file}')
    return surface_file


def convert_mgz2nii(FSmgz, bids_prefix='', keep=True):
    bsn = os.path.basename(FSmgz).split('.')
    nii_type = bsn.pop(0)
    spec = '_spec-FS'
    try:
        if len(bsn) > 1:
            spec = f'_spec-FS+{bsn[0]}'
    except:
        pass
    nifti = f'{os.path.dirname(FSmgz)}/{bids_prefix}_space-T2w{spec}_{nii_type}.nii.gz'
    fs2nii = f'mri_convert --in_type mgz --out_type nii {FSmgz} {nifti}'
    os.system(fs2nii)
    if os.path.isfile(nifti):
        if not keep:
            os.remove(FSmgz)
    print(f'{os.path.basename(FSmgz)} -> {os.path.basename(nifti)}')
    return nifti
