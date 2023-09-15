import os
import subprocess
import numpy as np
import nibabel as nib
from extract_ribbon import extract_ribbon


def process_raw_t2_subject(subject, session):

    # Define paths
    raw_dataset_folder = '/home/localadmin/Documents/PHD/data/MRI/Raw/Raw_NEXI-AIM1'
    preproc_dataset_folder = '/home/localadmin/Documents/PHD/data/MRI/NEXI-AIM1'
    outfold_main = f'{preproc_dataset_folder}'
    outf_sub = f'{outfold_main}/sub-{subject}'
    outf_ses = f'{outf_sub}/ses-{session}'
    bids_prefix = f'sub-{subject}_ses-{session}'

    # Recon_folder = "/home/localadmin/Documents/PHD/data/MRI/NEXI-AIM1/myelin_water_fraction"
    # subi = bids_prefix
    Recon_folder_subi = f'{outf_ses}/myelin_water_fraction'

    Data = "MET2.nii.gz"
    list_of_methods = ["X2", "L_curve"]
    reg_matrix = "I"

    # Merge data from each echo into a single file
    merged_t2_echo_folder = f'{raw_dataset_folder}/sub-{subject}/ses-{session}/nifti/myelin_water_fraction'
    merged_t2_echo_path = f'{merged_t2_echo_folder}/{Data}'
    if not os.path.isfile(merged_t2_echo_path):
        print("Merging data from each echo into a single file...")
        echo_number = 1
        echo_path = f'{merged_t2_echo_folder}/wip_mc_gse_1p8iso_76slc_32contrasts_e{echo_number}.nii.gz'
        basedata = np.expand_dims(nib.load(echo_path).get_fdata(), axis=-1)
        aff, hdr = nib.load(echo_path).affine, nib.load(echo_path).header
        for echo_number in range(2, 33):
            echo_path = f'{merged_t2_echo_folder}/wip_mc_gse_1p8iso_76slc_32contrasts_e{echo_number}.nii.gz'
            newdata = np.expand_dims(nib.load(echo_path).get_fdata(), axis=-1)
            basedata = np.concatenate([basedata, newdata], axis=3)
        merged_t2_echo_nii = nib.Nifti1Image(basedata, affine=aff, header=hdr)
        nib.save(merged_t2_echo_nii, merged_t2_echo_path)

    print(f"=========================== Subject: {subject}    Session : {session} ===========================")

    # Check if the subject's data exists
    if os.path.isfile(merged_t2_echo_path):
        # Create list of processed subjects
        # with open(f"{Recon_folder}/computed_subjects.txt", "a") as subjects_file:
        #     subjects_file.write(subi + "\n")

        os.makedirs(Recon_folder_subi, exist_ok=True)

        # Copy data to local folder using FSL (fslmaths)
        print("(1) Copy data to local folder")
        os.system(f'fslmaths {merged_t2_echo_path} {Recon_folder_subi}/MET2.nii.gz')

        # Remove Gibbs Ringing Artifacts using MRtrix3 (optional)
        print("(2) Remove Gibbs Ringing Artifacts, please wait...")
        os.system(f'mrdegibbs {Recon_folder_subi}/MET2.nii.gz {Recon_folder_subi}/MET2.nii.gz -force')

        # Brain extraction (BET) using FSL
        print("(3) BET, please wait...")
        os.system(f'fslmaths {Recon_folder_subi}/MET2.nii.gz -Tmean {Recon_folder_subi}/MET2_avg.nii.gz')
        os.system(f'bet {Recon_folder_subi}/MET2_avg.nii.gz {Recon_folder_subi}/MET2_mask -m -v -f 0.4')
        os.system(f'mv {Recon_folder_subi}/MET2_mask_mask.nii.gz {Recon_folder_subi}/mask.nii.gz')

        # Brain and Gray Matter Regions of Interest extraction using FastSurfer
        # print("(3) FastSurfer, please wait...")
        # os.system(f'python extract_ribbon.py -sub {subject} -ses {session}')

        # Estimate T2 spectra (for different methods)
        for Estimation_method in list_of_methods:
            print("=================== Estimation method:   ", Estimation_method, " ===================")
            print("(4) Non-parametric multicomponent T2 estimation, please wait...")
            os.system(f"python run_real_data_script.py --path_to_folder={Recon_folder_subi}/ --input='MET2.nii.gz' --mask='mask.nii.gz' --minTE=10.68 --nTE=32 --TR=1000 --FA_method='spline' --FA_smooth='yes' --denoise='TV' --reg_method={Estimation_method}  --reg_matrix={reg_matrix} --savefig='yes' --savefig_slice=35 --numcores=-1 --myelin_T2=40")

            # Bias-field correction (using FAST-FSL) of Total water content map
            print("(5) Bias-field correction (using FAST-FSL) of Total water content map...")
            if Estimation_method == "NNLS" or Estimation_method == "T2SPARC":
                recon_folder_name = "recon_all_{}".format(Estimation_method)
            else:
                recon_folder_name = "recon_all_{}-{}".format(Estimation_method, reg_matrix)

            # Estimate bias-field from the proton density map
            os.system(f"fast -t 3 -n 3 -H 0.1 -I 4 -l 20.0 -b -o {Recon_folder_subi}/{recon_folder_name}/TWC {Recon_folder_subi}/{recon_folder_name}/TWC")

            # Apply the correction to get the corrected map (i.e., corr = Raw/field-map)
            os.system(f"fslmaths {Recon_folder_subi}/{recon_folder_name}/TWC -div {Recon_folder_subi}/{recon_folder_name}/TWC_bias {Recon_folder_subi}/{recon_folder_name}/TWC")
    else:
        print("Error: Data {} does not exist".format(merged_t2_echo_path))
        # with open("Recon_folder/subjects_with_problems.txt", "a") as error_file:
        #     error_file.write(subi + "\n")

    fastsurfer_folder_subi = f'{outf_ses}/fastsurfer'
    t1w_parc = f'{fastsurfer_folder_subi}/mri/{bids_prefix}_space-T1w_spec-FS+DKTatlas+aseg_aparc.nii.gz'
    if not os.path.isfile(t1w_parc):
        print(f'Registering T1 ROI segmentation to T2 space for {subject} {session}')
        extract_ribbon(subject, session)
    else:
        print(f'Segmentation not found: {t1w_parc}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='T2 Myelin water fraction pipeline for NEXI data')
    parser.add_argument('-sub', '--subject', help='Subject', required=True)
    parser.add_argument('-ses', '--session', help='Session', required=True)
    args = parser.parse_args()

    process_raw_t2_subject(args.subject, args.session)
