import os
import shutil
import argparse
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

from dataset.dataset_test import MolTestDatasetWrapper
from utils.nt_xent import NTXentLoss

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_finetune.yaml', os.path.join(model_checkpoints_folder, 'config_finetune.yaml'))


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class FineTune(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        self.nt_xent_loss = NTXentLoss(device=self.device,
                                       batch_size=config['batch_size'],
                                       temperature=0.5,
                                       use_cosine_similarity=True)

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = current_time + '_' + config['task_name'] + '_' + config['dataset']['target']
        log_dir = os.path.join('finetune', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.dataset = dataset
        if config['dataset']['task'] == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        elif config['dataset']['task'] == 'regression':
            if self.config["task_name"] in ['qm7', 'qm8', 'qm9']:
                self.criterion = nn.L1Loss()
            else:
                self.criterion = nn.MSELoss()

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, data, n_iter):
        # get the prediction
        __, pred = model(data)  # [N,C]

        if self.config['dataset']['task'] == 'classification':
            loss = self.criterion(pred, data.y.flatten())
        elif self.config['dataset']['task'] == 'regression':
            if self.normalizer:
                loss = self.criterion(pred, self.normalizer.norm(data.y))
            else:
                loss = self.criterion(pred, data.y)

        return loss

    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        self.normalizer = None
        if self.config["task_name"] in ['qm7', 'qm9']:
            labels = []
            for d, __ in train_loader:
                labels.append(d.y)
            labels = torch.cat(labels)
            self.normalizer = Normalizer(labels)
            print(self.normalizer.mean, self.normalizer.std, labels.shape)
        if self.config["feat_type"] == 'plus':
            print("Plus the llm4sd and gnn feature ....")
            if self.config['model_type'] == 'gin':
                from models.ginet_finetune import GINet
                model = GINet(self.config['dataset']['task'], **self.config["model"]).to(self.device)
                model = self._load_pre_trained_weights(model)
            elif self.config['model_type'] == 'gcn':
                from models.gcn_finetune import GCN
                model = GCN(self.config['dataset']['task'], **self.config["model"]).to(self.device)
                model = self._load_pre_trained_weights(model)
        elif self.config["feat_type"] == 'dir_concat':
            print("Directly Concat the llm4sd and gnn feature ....")
            if self.config['model_type'] == 'gin':
                from models.ginet_finetune_dir_concat import GINet
                model = GINet(self.config['dataset']['task'], **self.config["model"]).to(self.device)
                model = self._load_pre_trained_weights(model)
            elif self.config['model_type'] == 'gcn':
                from models.gcn_finetune_dir_concat import GCN
                model = GCN(self.config['dataset']['task'], **self.config["model"]).to(self.device)
                model = self._load_pre_trained_weights(model)
        else:
            raise ValueError('Undefined feat_type! plus or concat')

        layer_list = []
        for name, param in model.named_parameters():
            if 'pred_head' in name:
                print(name, param.requires_grad)
                layer_list.append(name)

        params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(
            map(lambda x: x[1], list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

        optimizer = torch.optim.Adam(
            [{'params': base_params, 'lr': self.config['init_base_lr']}, {'params': params}],
            self.config['init_lr'], weight_decay=eval(self.config['weight_decay'])
        )

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level='O2', keep_batchnorm_fp32=True
            )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_rgr = np.inf
        best_valid_cls = 0

        for epoch_counter in range(self.config['epochs']):
            for bn, data in enumerate(train_loader):
                optimizer.zero_grad()

                data = data.to(self.device)
                loss = self._step(model, data, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print(epoch_counter, bn, loss.item())

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                if self.config['dataset']['task'] == 'classification':
                    valid_loss, valid_cls = self._validate(model, valid_loader)
                    if valid_cls > best_valid_cls:
                        # save the model weights
                        best_valid_cls = valid_cls
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                elif self.config['dataset']['task'] == 'regression':
                    valid_loss, valid_rgr = self._validate(model, valid_loader)
                    if valid_rgr < best_valid_rgr:
                        # save the model weights
                        best_valid_rgr = valid_rgr
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

        self._test(model, test_loader)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=self.device)
            # model.load_state_dict(state_dict)
            model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                __, pred = model(data)
                loss = self._step(model, data, bn)

                valid_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            valid_loss /= num_data

        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                mae = mean_absolute_error(labels, predictions)
                print('Validation loss:', valid_loss, 'MAE:', mae)
                return valid_loss, mae
            else:
                rmse = mean_squared_error(labels, predictions, squared=False)
                print('Validation loss:', valid_loss, 'RMSE:', rmse)
                return valid_loss, rmse

        elif self.config['dataset']['task'] == 'classification':
            predictions = np.array(predictions)
            labels = np.array(labels)
            roc_auc = roc_auc_score(labels, predictions[:, 1])
            print('Validation loss:', valid_loss, 'ROC AUC:', roc_auc)
            return valid_loss, roc_auc

    def _test(self, model, test_loader):
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)

                __, pred = model(data)
                loss = self._step(model, data, bn)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            test_loss /= num_data

        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                self.mae = mean_absolute_error(labels, predictions)
                print('Test loss:', test_loss, 'Test MAE:', self.mae)
            else:
                self.rmse = mean_squared_error(labels, predictions, squared=False)
                print('Test loss:', test_loss, 'Test RMSE:', self.rmse)

        elif self.config['dataset']['task'] == 'classification':
            predictions = np.array(predictions)
            labels = np.array(labels)
            self.roc_auc = roc_auc_score(labels, predictions[:, 1])
            print('Test loss:', test_loss, 'Test ROC AUC:', self.roc_auc)


def main(config):
    dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])
    train_l, valid_l, test_l = dataset.get_data_loaders()
    example_data = next(iter(train_l))
    config['model']['llm4sd_x_dim'] = example_data.llm4sd_x.shape[1]

    fine_tune = FineTune(dataset, config)
    fine_tune.train()

    if config['dataset']['task'] == 'classification':
        return fine_tune.roc_auc
    if config['dataset']['task'] == 'regression':
        if config['task_name'] in ['qm7', 'qm8', 'qm9']:
            return fine_tune.mae
        else:
            return fine_tune.rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, default='scaffold_datasets', help="load train/valid/test dataset")
    parser.add_argument('--dataset', type=str, default='freesolv', help='dataset name')
    parser.add_argument('--subtask', type=str, default='', help='subtask of tox21/sider/qm9 dataset')
    parser.add_argument('--model', type=str, default='galactica-6.7b', help='LLM model')
    parser.add_argument('--knowledge_type', type=str, default='all', help='synthesize/inference/all')
    parser.add_argument('--num_samples', type=int, default=30, help='number of sample lists (30/50) for inference')
    parser.add_argument('--feat_type', type=str, default='dir_concat', help='dir_concat or plus')
    parser.add_argument('--drop_ratio', type=float, default=0.3)
    args = parser.parse_args()

    config = yaml.load(open("config_finetune.yaml", "r"), Loader=yaml.FullLoader)
    config['dataset']["dataset_name"] = args.dataset.lower()
    config['dataset']["subtask"] = args.subtask.lower()
    config['dataset']["model"] = args.model
    config['dataset']["knowledge_type"] = args.knowledge_type
    config['dataset']["num_samples"] = args.num_samples
    config['model']['drop_ratio'] = args.drop_ratio
    config['feat_type'] = args.feat_type

    if args.dataset in ["alpha", "c_v", "Delta_epsilon", "epsilon_HOMO",
                        "epsilon_LUMO", "G", "H", "mu", "R^2", "U_0", "U", "ZPVE"]:
        file_folder = os.path.join(args.dataset_folder, 'qm9')
    else:
        file_folder = os.path.join(args.dataset_folder, args.dataset.lower())

    if args.subtask == "":
        train_file_name = args.dataset.lower() + '_train.csv'
        valid_file_name = args.dataset.lower() + '_valid.csv'
        test_file_name = args.dataset.lower() + '_test.csv'
    else:
        train_file_name = args.subtask.lower() + '_train.csv'
        valid_file_name = args.subtask.lower() + '_valid.csv'
        test_file_name = args.subtask.lower() + '_test.csv'
    train_file_path = os.path.join(file_folder, train_file_name)
    valid_file_path = os.path.join(file_folder, valid_file_name)
    test_file_path = os.path.join(file_folder, test_file_name)

    config['dataset']['train_data_path'] = train_file_path
    config['dataset']['valid_data_path'] = valid_file_path
    config['dataset']['test_data_path'] = test_file_path

    config['task_name'] = args.dataset.lower()

    if config['task_name'] in ["bbbp", "tox21", "clintox",
                               "hiv", "bace", "sider"]:
        config['dataset']['task'] = 'classification'
    else:
        config['dataset']['task'] = 'regression'

    if config['task_name'] == 'bbbp':
        target_list = ["p_np"]

    elif config['task_name'] == 'tox21':
        target_list = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]

    elif config['task_name'] == 'clintox':
        target_list = ['CT_TOX']

    elif config['task_name'] == 'hiv':
        target_list = ["HIV_active"]

    elif config['task_name'] == 'bace':
        target_list = ["Class"]

    elif config['task_name'] == 'sider':
        target_list = [
            "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues",
            "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders",
            "Gastrointestinal disorders", "Social circumstances", "Immune system disorders",
            "Reproductive system and breast disorders",
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
            "General disorders and administration site conditions", "Endocrine disorders",
            "Surgical and medical procedures", "Vascular disorders",
            "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders",
            "Congenital, familial and genetic disorders", "Infections and infestations",
            "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders",
            "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions",
            "Ear and labyrinth disorders", "Cardiac disorders",
            "Nervous system disorders", "Injury, poisoning and procedural complications"
        ]

    elif config['task_name'] == 'freesolv':
        target_list = ["expt"]

    elif config['task_name'] == 'esol':
        target_list = ["measured log solubility in mols per litre"]

    elif config['task_name'] == 'lipophilicity':
        target_list = ["exp"]

    elif config['task_name'] == 'qm9':
        target_list = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv']

    else:
        raise ValueError('Undefined downstream task!')

    print(config)

    headers = ['target', config['feat_type'], 'knowledge_type']

    results_list = []
    for target in target_list:
        config['dataset']['target'] = target
        result = main(config)

        if args.knowledge_type == "synthesize":
            results_list.append([target, result, args.knowledge_type])
        else:
            results_list.append([target, result, args.knowledge_type, args.num_samples])
            headers.append('num_samples')

    if args.knowledge_type != 'synthesize':
        num_sample = f'{args.num_samples}_'
    else:
        num_sample = ''

    # Output reseults
    drop_ratio = config['model']['drop_ratio']
    results_folder = f"experiments_multiView_{config['feat_type']}_{drop_ratio}"
    os.makedirs(results_folder, exist_ok=True)

    file_path = '{}/{}_{}_{}_{}_{}finetune.csv'.format(results_folder, config['fine_tune_from'],
                                                       config['task_name'], args.model,
                                                       args.knowledge_type, num_sample)

    # Check if file exists
    if not os.path.isfile(file_path):
        # Write header and data
        df = pd.DataFrame(results_list, columns=headers)
        df.to_csv(file_path, mode='w', index=False, header=True)
    else:
        # Append data without header
        df = pd.DataFrame(results_list)
        df.to_csv(file_path, mode='a', index=False, header=False)