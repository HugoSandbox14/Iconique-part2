import mne
from pathlib import Path
from parcellation_tools import Parcellation , Forward



fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
mne.datasets.fetch_aparc_sub_parcellation(subjects_dir=fs_dir)

subjects_dir = Path(fs_dir).parent
subject='fsaverage'

model = mne.make_bem_model(subject = subject, subjects_dir = subjects_dir)
bem = mne.make_bem_solution(model)

labels = mne.read_labels_from_annot(
    subject=subject,
    parc='aparc',
    subjects_dir=subjects_dir
)

parcellation_obj = Parcellation(subjects_dir,subject)
forward_obj = Forward(subjects_dir,subject)
G = forward_obj.G_mean_reference