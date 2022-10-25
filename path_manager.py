# %%pycodestyle
from dataclasses import dataclass
from dataclasses import field
import os
import pandas as pd
import pickle


@dataclass
class PathManager:
    competition_path: str
    preprocessing_trial: int
    models_trial: int

    def __post_init__(self):
        self.data_root_path = os.path.join(self.competition_path, 'Data')
        self.models_root_path = os.path.join(self.competition_path, 'Models')

        self.data_trial_path = os.path.join(
            self.data_root_path,
            f'preproc_trial_{self.preprocessing_trial}'
        )
        self.models_trial_path = os.path.join(
            self.models_root_path,
            f'trial_{self.models_trial}'
        )

    @property
    def train_path(self):
        return os.path.join(self.data_root_path, 'train.csv')

    @property
    def test_path(self):
        return os.path.join(self.data_root_path, 'test.csv')

    def _create_path(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        else:
            print(f'{path} already exists')

    def save_to_csv(self, array_, path_, file_name_):
        df = pd.DataFrame(array_)
        df.to_csv(os.path.join(path_, file_name_))

    def setup_paths(self):
        # precondition
        assert os.path.isdir(self.data_root_path), print(self.data_root_path)
        assert os.path.isdir(self.models_root_path), \
            print(self.models_root_path)

        self._create_path(self.data_trial_path)
        self._create_path(self.models_trial_path)

    def save_models(self, sklearn_models_dict_):
        '''
        For each model in the dictionary, create a folder.
        Save pickle file there. If model (or folder) already exists -
        overrides all files
        '''
        for model_name, sklearn_model in sklearn_models_dict_.items():
            folder_path = os.path.join(
                self.models_trial_path, model_name
            )
            self._create_path(folder_path)
            model_filename = model_name + '.sav'
            pickle.dump(
                sklearn_model,
                open(
                    os.path.join(folder_path, model_filename),
                    'w+b'
                ),
            )

    def load_models(self, models_subfolders_=[]):
        '''
        1. Goes to self.models_trial_path
        2. If models_names=[] - downloads all
           models from their subfolders.
        3. models_names must contain names of subfolders on
           google drive!
        '''

        # precondition
        assert os.path.isdir(self.models_trial_path), \
            print(self.models_trial_path)

        available_models_subfolders = os.listdir(self.models_trial_path)
        if models_subfolders_:
            models_to_download = models_subfolders_
        else:
            # Download everything
            models_to_download = available_models_subfolders

        res = {}
        for model_subfolder in models_to_download:
            # file name coincides with folder name
            file_name = f'{model_subfolder}.sav'
            full_path = os.path.join(
                self.models_trial_path,
                model_subfolder,
                file_name
            )

            loaded_model = pickle.load(
                open(full_path, 'rb')
            )
            res[model_subfolder] = loaded_model
        return res
