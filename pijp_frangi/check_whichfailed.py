import os

input_dir = '/m/Researchers/SerenaT/deeppvs/for_nnunet/groundtruth_rawimgs'
output_dir = '/m/Researchers/SerenaT/deeppvs/for_nnunet/gt_mcpvs_preprocessed3'

failed = []
for subject in os.listdir(input_dir):
    subj_out = os.path.join(output_dir, subject)
    # check for the final output file
    norm_file = [f for f in os.listdir(subj_out) if f.endswith('-T1bcbrainmask_norm.nii.gz')] if os.path.isdir(subj_out) else []
    if not norm_file:
        failed.append(subject)

print(f"Failed/incomplete subjects: {len(failed)}")
with open('./redo_subjects.txt', 'w') as f:
    for s in failed:
        print(s)
        f.write(f'{s}\n')
