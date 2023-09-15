import os
import nibabel as nib


def extract_ribbon(subject, session):
    # Define paths
    # FastSurferBin = '/home/localadmin/GitHub/FastSurfer'
    # cores = 10
    raw_dataset_folder = '/home/localadmin/Documents/PHD/data/MRI/Raw/Raw_NEXI-AIM1'
    preproc_dataset_folder = '/home/localadmin/Documents/PHD/data/MRI/NEXI-AIM1'
    outfold_main = f'{preproc_dataset_folder}'
    outf_sub = f'{outfold_main}/sub-{subject}'
    if not os.path.isdir(outf_sub):
        os.mkdir(outf_sub)
    outf_ses = f'{outf_sub}/ses-{session}'
    if not os.path.isdir(outf_ses):
        os.mkdir(outf_ses)
    bids_prefix = f'sub-{subject}_ses-{session}'
    # myelin_t2_folder = f'{outf_ses}/myelin_water_fraction'
    # Define Fastsurfer T2 folder
    # fastsurfer_t2_folder = f'{myelin_t2_folder}/fastsurfer'
    # if not os.path.isdir(fastsurfer_t2_folder):
    #     os.mkdir(fastsurfer_t2_folder)

    # Find first volume of multi-echo T2
    t2_vol0 = f'{outf_ses}/myelin_water_fraction/MET2_volume0.nii.gz'
    # Find multi-echo T2
    t2_me = f'{outf_ses}/myelin_water_fraction/MET2.nii.gz'
    t2_me_nii = nib.load(t2_me)
    aff_t2_me, hdr_t2_me = t2_me_nii.affine, t2_me_nii.header

    if not os.path.isfile(t2_vol0):
        t2_me_data = t2_me_nii.get_fdata()
        t2_vol0_data = t2_me_data[:, :, :, 0]
        t2_vol0_nii = nib.Nifti1Image(t2_vol0_data, affine=aff_t2_me, header=hdr_t2_me)
        nib.save(t2_vol0_nii, t2_vol0)
    # Volumetric segmentation
    # print(f'Starting FastSurfer segmentation for {subject} {session}')
    # fastsurfer_volumetric_segmentation(myelin_t2_folder, t2_vol0, FastSurferBin, cores)
    # Convert MGZ to nifti
    # mgz_file = f'{outf_ses}/myelin_water_fraction/fastsurfer/mri/aparc.DKTatlas+aseg.deep.mgz'
    # convert_mgz2nii(mgz_file, bids_prefix=bids_prefix, keep=True)

    t1brain = f'{raw_dataset_folder}/sub-{subject}/ses-{session}/nifti/t1_mprage_1iso_IJ.nii.gz'
    outfold = f'{outf_ses}/myelin_water_fraction/{bids_prefix}'

    # same brain no warp
    affine_t2w2t1w = f"{outfold}_from-t2w_to-t1w_0GenericAffine.mat"
    # Warped : t2w brain in t1 space
    t2w_to_t1w_filename = f"{outfold}_space-t1w_t2w.nii.gz"
    # InverseWarped : T1 brain in t2w space
    t1w_to_t2w_filename = f"{outfold}_space-t2w_tw1.nii.gz"
    if not os.path.isfile(t1w_to_t2w_filename):
        quickAffine(t1brain, t2_vol0, f'{outfold}_from-t2w_to-t1w_', debug=False,
                    metric='MI', conv=13, itsc=9, sz=[0.1, 0.1])
        os.rename(f"{outfold}_from-t2w_to-t1w_Warped.nii.gz", t2w_to_t1w_filename)
        os.rename(f"{outfold}_from-t2w_to-t1w_InverseWarped.nii.gz", t1w_to_t2w_filename)

    # Find T1 parcellation
    parc_T1w_filename = f'{outf_ses}/fastsurfer/mri/{bids_prefix}_space-T1w_spec-FS+DKTatlas+aseg_aparc.nii.gz'
    assert os.path.isfile(parc_T1w_filename), f'File {parc_T1w_filename} not found'

    # Transform T1 to t2w space
    transform_file = affine_t2w2t1w
    assert os.path.isfile(transform_file), f'File {transform_file} not found'
    transform = _invertAffine(transform_file)
    parc_t2w_filename = f'{outf_ses}/fastsurfer/mri/{bids_prefix}_space-T2w_spec-FS+DKTatlas+aseg_aparc.nii.gz'
    reference = t2_vol0
    _ANTs_ApplyTransform(parc_T1w_filename,
                         parc_t2w_filename,
                         reference,
                   transform)

    # Define cortical ribbon mask
    parc_t2w_nifti = nib.load(parc_t2w_filename)
    parc_t2w = parc_t2w_nifti.get_fdata()
    cort_rib_t2w = (parc_t2w >= 1000).astype(int)
    # aff, hdr = parc_t2w_nifti.affine, parc_t2w_nifti.header
    cort_rib_t2w_nifti = nib.Nifti1Image(cort_rib_t2w, affine=aff_t2_me, header=hdr_t2_me)
    cort_rib_t2w_filename = f'{outf_ses}/myelin_water_fraction/{bids_prefix}_space-T2w_desc-aparc_ribbon.nii.gz'
    nib.save(cort_rib_t2w_nifti, cort_rib_t2w_filename)

    # Re-define cortical ribbon ROI
    cort_rib_roi_t2w = parc_t2w
    cort_rib_roi_t2w_nifti = nib.Nifti1Image(cort_rib_roi_t2w, affine=aff_t2_me, header=hdr_t2_me)
    cort_rib_roi_t2w_filename = f'{outf_ses}/myelin_water_fraction/{bids_prefix}_space-T2w_desc-aparc_ribbon_roi.nii.gz'
    nib.save(cort_rib_roi_t2w_nifti, cort_rib_roi_t2w_filename)


def quickAffine(f, m, o, debug=False, metric='MI', conv='6', itsc=1, sz=None):
    """antsQuickReg: f=fixed, m=moving, o=outPrefix, conv=convergence thr, itsc=base number itererations * scale"""
    if metric == 'CC':
        rb = 1  # radius 1
    else:
        rb = 32  # bins
    if sz is None:
        sz = [0.1, 0.1]
    l0, l1, l2, l3, rep = 1000 * itsc, 500 * itsc, 250 * itsc, 100 * itsc, 10 * int(1 + itsc / 2)
    call = [f'antsRegistration ',
            f' --verbose {int(debug)} ',
            f' --dimensionality 3 ',
            f' --float 0 ',
            f' --collapse-output-transforms 1',
            f' --output [ {o},{o}Warped.nii.gz,{o}InverseWarped.nii.gz ] ',
            f' --interpolation Linear --use-histogram-matching 0 --winsorize-image-intensities [ 0.005,0.995 ] ',
            f' --initial-moving-transform [ {f},{m},1 ] ',
            f' --transform Rigid[ {sz[0]} ] ',
            f' --metric {metric}[ {f},{m},1,{rb},Regular,0.5 ] ',
            f' --convergence [ {l0}x{l1}x{l2}x{l3},1e-{conv},{rep} ] ',
            f' --shrink-factors 8x4x2x1 ',
            f' --smoothing-sigmas 3x2x1x0vox ',
            f' --transform Affine[ {sz[1]} ] ',
            f' --metric {metric}[ {f},{m},1,{rb},Regular,0.5 ] ',
            f' --convergence [ {l0}x{l1}x{l2}x{l3},1e-{conv},{rep} ] ',
            f' --shrink-factors 8x4x2x1 ',
            f' --smoothing-sigmas 3x2x1x0vox']
    if debug:
        print(call)
    return os.system(' '.join(call))


def _invertAffine(affine):
    return f'[{affine}, 1]'


def _ANTs_ApplyTransform(indwi, warpeddwi, reference=None, *args, interp=1, debug=False, antsbin=''):
    """
        Apply ANTs transforms to an input image and generate a warped output image.

        Args:
            indwi (str): Path to the input DWI image.
            warpeddwi (str): Path to save the warped DWI image.
            reference (str, optional): Path to the reference image. Defaults to None.
            *args: Variable number of transform paths to be applied.
            interp (int, optional): Interpolation type. 0: Linear, 1: NearestNeighbor, 2: GenericLabel. Defaults to 1.
            debug (bool, optional): Print the command if True. Defaults to False.
            antsbin (str, optional): Path to the ANTs command-line tools. Defaults to ''.

        Returns:
            int: The return code of the ANTs command.

        Raises:
            FileNotFoundError: If the input or reference image file is not found.

        Notes:
            - Requires ANTs (Advanced Normalization Tools) command-line tools to be installed.
            - The `antsbin` argument should be provided with the path to the ANTs command-line tools.
            - The `interp` argument accepts values 0, 1, or 2 to specify the interpolation type.
            - If `debug` is True, the generated ANTs command will be printed.
    """

    e = '-e 3 ' if len(nib.load(indwi).shape) == 4 else ''
    interpolation = {0: 'Linear', 1: 'NearestNeighbor', 2: 'GenericLabel'}
    reference_option = f'-r {reference}' if (reference is not None) else ''
    spec = f'--default-value 0 --interpolation {interpolation[interp]} '
    transform_options = ' '.join([f'-t {transform}' for transform in args])
    command = f'{antsbin}antsApplyTransforms -d 3 {e}-i {indwi} -o {warpeddwi} {reference_option} {transform_options} {spec}'

    if debug:
        print(f'\n{command}\n')

    return os.system(command)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='NEXI MET2 Pre-processing pipeline')
    parser.add_argument('-sub', '--subject', help='Subject', required=True)
    parser.add_argument('-ses', '--session', help='Session', required=True)
    args = parser.parse_args()

    # Define paths
    preproc_dataset_folder = '/home/localadmin/Documents/PHD/data/MRI/NEXI-AIM1'
    outfold_main = f'{preproc_dataset_folder}'
    outf_sub = f'{outfold_main}/sub-{args.subject}'
    outf_ses = f'{outf_sub}/ses-{args.session}'
    bids_prefix = f'sub-{args.subject}_ses-{args.session}'
    # Find T2 parcellation
    parc_T2w_filename = f'{outf_ses}/myelin_water_fraction/fastsurfer/mri/{bids_prefix}_space-T2w_spec-FS+DKTatlas+aseg_aparc.nii.gz'
    # Extract ribbon or transform parcellation once preprocessing is done
    extract_ribbon(args.subject, args.session)
