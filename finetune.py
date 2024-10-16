import os
import shutil
import argparse
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import subprocess

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

from dataset.dataset_test import MolTestDatasetWrapper

from torch_geometric.data import DataLoader
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import json
from sklearn.metrics import roc_auc_score, mean_absolute_error

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
        __, pred_molclr, pred_cat, pred_llm = model(data)  # [N,C]

        if self.config['dataset']['task'] == 'classification':
            loss_molclr = self.criterion(pred_molclr, data.y.flatten())
            loss_cat = self.criterion(pred_cat, data.y.flatten())
            loss_llm = self.criterion(pred_llm, data.y.flatten())
        elif self.config['dataset']['task'] == 'regression':
            if self.normalizer:
                loss_molclr = self.criterion(pred_molclr, self.normalizer.norm(data.y))
                loss_cat = self.criterion(pred_cat, self.normalizer.norm(data.y))
                loss_llm = self.criterion(pred_llm, self.normalizer.norm(data.y))
            else:
                loss_molclr = self.criterion(pred_molclr, data.y)
                loss_cat = self.criterion(pred_cat, data.y)
                loss_llm = self.criterion(pred_llm, data.y)

        return loss_molclr, loss_cat, loss_llm

    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        self.normalizer = None
        if self.config["task_name"] in ['qm7', 'qm9']:
            labels = []
            for d in train_loader.dataset:
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
        elif self.config["feat_type"] == 'concat':
            print("Concat the llm4sd (512) and gnn feature ....")
            if self.config['model_type'] == 'gin':
                from models.ginet_finetune_concat import GINet
                model = GINet(self.config['dataset']['task'], **self.config["model"]).to(self.device)
                model = self._load_pre_trained_weights(model)
            elif self.config['model_type'] == 'gcn':
                from models.gcn_finetune_concat import GCN
                model = GCN(self.config['dataset']['task'], **self.config["model"]).to(self.device)
                model = self._load_pre_trained_weights(model)
        else:
            raise ValueError('Undefined feat_type! plus or concat')

        # Separate the parameters for the GNN + pred_molclr and pred_cat
        gnn_params = []
        pred_molclr_params = []
        pred_cat_params = []
        pred_llm_params = []

        for name, param in model.named_parameters():
            if 'pred_head_1' in name:
                pred_molclr_params.append(param)
            elif 'pred_head_2' in name:
                pred_cat_params.append(param)
            elif 'pred_head_3' in name:
                pred_llm_params.append(param)
            else:
                gnn_params.append(param)

        optimizer_gnn_molclr = torch.optim.Adam(
            [{'params': gnn_params, 'lr': self.config['init_base_lr']}, {'params': pred_molclr_params}],
            self.config['init_lr'], weight_decay=eval(self.config['weight_decay'])
        )

        optimizer_cat = torch.optim.Adam(
            pred_cat_params, self.config['init_lr'], weight_decay=eval(self.config['weight_decay'])
        )
        optimizer_llm = torch.optim.Adam(
            pred_llm_params, self.config['init_lr'], weight_decay=eval(self.config['weight_decay'])
        )

        if apex_support and self.config['fp16_precision']:
            model, optimizer_gnn_molclr = amp.initialize(
                model, optimizer_gnn_molclr, opt_level='O2', keep_batchnorm_fp32=True
            )
            optimizer_cat = amp.initialize(
                model, optimizer_cat, opt_level='O2', keep_batchnorm_fp32=True
            )[1]
            optimizer_llm = amp.initialize(
                model, optimizer_llm, opt_level='O2', keep_batchnorm_fp32=True
            )[1]

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss_molclr = np.inf
        best_valid_loss_cat = np.inf
        best_valid_loss_llm = np.inf
        best_valid_rgr_molclr = np.inf
        best_valid_rgr_cat = np.inf
        best_valid_rgr_llm = np.inf
        best_valid_cls_molclr = 0
        best_valid_cls_cat = 0
        best_valid_cls_llm = 0

        torch.autograd.set_detect_anomaly(True)

        for epoch_counter in range(self.config['epochs']):
            for bn, data in enumerate(train_loader):
                optimizer_gnn_molclr.zero_grad()
                optimizer_cat.zero_grad()
                optimizer_llm.zero_grad()

                data = data.to(self.device)
                loss_molclr, loss_cat, loss_llm = self._step(model, data, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss_molclr', loss_molclr.item(), global_step=n_iter)
                    self.writer.add_scalar('train_loss_cat', loss_cat.item(), global_step=n_iter)
                    self.writer.add_scalar('train_loss_llm', loss_llm.item(), global_step=n_iter)
                    print(epoch_counter, bn, 'Loss LLM:', loss_llm.item(), 'Loss Molclr:', loss_molclr.item(),
                          'Loss Cat:', loss_cat.item())

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss_molclr, optimizer_gnn_molclr) as scaled_loss:
                        scaled_loss.backward(retain_graph=True)

                    with amp.scale_loss(loss_cat, optimizer_cat) as scaled_loss:
                        scaled_loss.backward()
                    with amp.scale_loss(loss_llm, optimizer_llm) as scaled_loss:
                        scaled_loss.backward()
                    optimizer_gnn_molclr.step()
                    optimizer_cat.step()
                    optimizer_llm.step()
                else:
                    # Backward pass and optimization for GNN + pred_molclr
                    loss_molclr.backward(retain_graph=True)
                    # Backward pass and optimization for pred_cat
                    loss_cat.backward()
                    loss_llm.backward()
                    optimizer_gnn_molclr.step()
                    optimizer_cat.step()
                    optimizer_llm.step()

                n_iter += 1

            # Validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                if self.config['dataset']['task'] == 'classification':
                    valid_loss_molclr, valid_cls_molclr, valid_loss_cat, valid_cls_cat, valid_loss_llm, valid_cls_llm = self._validate(
                        model, valid_loader)
                    if valid_cls_molclr > best_valid_cls_molclr:
                        best_valid_cls_molclr = valid_cls_molclr
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_molclr.pth'))
                    if valid_cls_cat > best_valid_cls_cat:
                        best_valid_cls_cat = valid_cls_cat
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_cat.pth'))
                    if valid_cls_llm > best_valid_cls_llm:
                        best_valid_cls_llm = valid_cls_llm
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_llm.pth'))
                elif self.config['dataset']['task'] == 'regression':
                    valid_loss_molclr, valid_rgr_molclr, valid_loss_cat, valid_rgr_cat, valid_loss_llm, valid_rgr_llm = self._validate(
                        model, valid_loader)
                    if valid_rgr_molclr < best_valid_rgr_molclr:
                        best_valid_rgr_molclr = valid_rgr_molclr
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_molclr.pth'))
                    if valid_rgr_cat < best_valid_rgr_cat:
                        best_valid_rgr_cat = valid_rgr_cat
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_cat.pth'))
                    if valid_rgr_llm < best_valid_rgr_llm:
                        best_valid_rgr_llm = valid_rgr_llm
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_llm.pth'))

                self.writer.add_scalar('validation_loss_molclr', valid_loss_molclr, global_step=valid_n_iter)
                self.writer.add_scalar('validation_loss_cat', valid_loss_cat, global_step=valid_n_iter)
                self.writer.add_scalar('validation_loss_llm', valid_loss_llm, global_step=valid_n_iter)
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
        predictions_molclr = []
        predictions_cat = []
        predictions_llm = []
        labels = []
        with torch.no_grad():
            model.eval()

            valid_loss_molclr = 0.0
            valid_loss_cat = 0.0
            valid_loss_llm = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                __, pred_molclr, pred_cat, pred_llm = model(data)
                loss_molclr, loss_cat, loss_llm = self._step(model, data, bn)

                valid_loss_molclr = valid_loss_molclr + (loss_molclr.item() * data.y.size(0))
                valid_loss_cat = valid_loss_cat + (loss_cat.item() * data.y.size(0))
                valid_loss_llm = valid_loss_llm + (loss_llm.item() * data.y.size(0))
                num_data = num_data + data.y.size(0)

                if self.normalizer:
                    pred_molclr = self.normalizer.denorm(pred_molclr)
                    pred_cat = self.normalizer.denorm(pred_cat)
                    pred_llm = self.normalizer.denorm(pred_llm)

                if self.config['dataset']['task'] == 'classification':
                    pred_molclr = F.softmax(pred_molclr, dim=-1)
                    pred_cat = F.softmax(pred_cat, dim=-1)
                    pred_llm = F.softmax(pred_llm, dim=-1)

                if self.device == 'cpu':
                    predictions_molclr.extend(pred_molclr.detach().numpy())
                    predictions_cat.extend(pred_cat.detach().numpy())
                    predictions_llm.extend(pred_llm.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions_molclr.extend(pred_molclr.cpu().detach().numpy())
                    predictions_cat.extend(pred_cat.cpu().detach().numpy())
                    predictions_llm.extend(pred_llm.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            valid_loss_molclr /= num_data
            valid_loss_cat /= num_data
            valid_loss_llm /= num_data

        model.train()

        predictions_molclr = np.array(predictions_molclr)
        predictions_cat = np.array(predictions_cat)
        predictions_llm = np.array(predictions_llm)
        labels = np.array(labels)

        if self.config['dataset']['task'] == 'regression':
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                mae_molclr = mean_absolute_error(labels, predictions_molclr)
                mae_cat = mean_absolute_error(labels, predictions_cat)
                mae_llm = mean_absolute_error(labels, predictions_llm)
                print('Validation loss molclr:', valid_loss_molclr, 'MAE molclr:', mae_molclr,
                      '\nValidation loss cat:', valid_loss_cat, 'MAE cat:', mae_cat,
                      '\nValidation loss llm:', valid_loss_llm, 'MAE llm:', mae_llm)
                print('-------------------------------------------------------------------------')
                return valid_loss_molclr, mae_molclr, valid_loss_cat, mae_cat, valid_loss_llm, mae_llm
            else:
                rmse_molclr = mean_squared_error(labels, predictions_molclr, squared=False)
                rmse_cat = mean_squared_error(labels, predictions_cat, squared=False)
                rmse_llm = mean_squared_error(labels, predictions_llm, squared=False)
                print('Validation loss molclr:', valid_loss_molclr, 'RMSE molclr:', rmse_molclr,
                      '\nValidation loss cat:', valid_loss_cat, 'RMSE cat:', rmse_cat,
                      '\nValidation loss llm:', valid_loss_llm, 'RMSE llm:', rmse_llm)
                print('-------------------------------------------------------------------------')
                return valid_loss_molclr, rmse_molclr, valid_loss_cat, rmse_cat, valid_loss_llm, rmse_llm

        elif self.config['dataset']['task'] == 'classification':
            roc_auc_molclr = roc_auc_score(labels, predictions_molclr[:, 1])
            roc_auc_cat = roc_auc_score(labels, predictions_cat[:, 1])
            roc_auc_llm = roc_auc_score(labels, predictions_llm[:, 1])
            print('Validation loss molclr:', valid_loss_molclr, 'ROC AUC molclr:', roc_auc_molclr,
                  '\nValidation loss cat:', valid_loss_cat, 'ROC AUC cat:', roc_auc_cat,
                  '\nValidation loss llm:', valid_loss_llm, 'ROC AUC llm:', roc_auc_llm)
            print('-------------------------------------------------------------------------')
        return valid_loss_molclr, roc_auc_molclr, valid_loss_cat, roc_auc_cat, valid_loss_llm, roc_auc_llm

    def _test(self, model, test_loader):
        model_path_molclr = os.path.join(self.writer.log_dir, 'checkpoints', 'model_molclr.pth')
        model_path_cat = os.path.join(self.writer.log_dir, 'checkpoints', 'model_cat.pth')
        model_path_llm = os.path.join(self.writer.log_dir, 'checkpoints', 'model_llm.pth')

        if os.path.exists(model_path_molclr):
            state_dict = torch.load(model_path_molclr, map_location=self.device)
            model.load_state_dict(state_dict)
            print("Loaded MolCLR trained model with success.")
        else:
            print(f"Model checkpoint {model_path_molclr} not found.")

        # Assuming you have a separate model instance for `cat`, or load state dicts as needed.
        if os.path.exists(model_path_cat):
            state_dict = torch.load(model_path_cat, map_location=self.device)
            model.load_state_dict(state_dict)
            print("Loaded Cat trained model with success.")
        else:
            print(f"Model checkpoint {model_path_cat} not found.")

        if os.path.exists(model_path_llm):
            state_dict = torch.load(model_path_molclr, map_location=self.device)
            model.load_state_dict(state_dict)
            print("Loaded LLM trained model with success.")
        else:
            print(f"Model checkpoint {model_path_llm} not found.")

        # test steps
        predictions_molclr = []
        predictions_cat = []
        predictions_llm = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss_molclr = 0.0
            test_loss_cat = 0.0
            test_loss_llm = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)

                __, pred_molclr, pred_cat, pred_llm = model(data)
                loss_molclr, loss_cat, loss_llm = self._step(model, data, bn)

                test_loss_molclr = test_loss_molclr + (loss_molclr.item() * data.y.size(0))
                test_loss_cat = test_loss_cat + (loss_cat.item() * data.y.size(0))
                test_loss_llm = test_loss_llm + (loss_llm.item() * data.y.size(0))
                num_data = num_data + data.y.size(0)

                if self.normalizer:
                    pred_molclr = self.normalizer.denorm(pred_molclr)
                    pred_cat = self.normalizer.denorm(pred_cat)
                    pred_llm = self.normalizer.denorm(pred_llm)

                if self.config['dataset']['task'] == 'classification':
                    pred_molclr = F.softmax(pred_molclr, dim=-1)
                    pred_cat = F.softmax(pred_cat, dim=-1)
                    pred_llm = F.softmax(pred_llm, dim=-1)

                if self.device == 'cpu':
                    predictions_molclr.extend(pred_molclr.detach().numpy())
                    predictions_cat.extend(pred_cat.detach().numpy())
                    predictions_llm.extend(pred_llm.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions_molclr.extend(pred_molclr.cpu().detach().numpy())
                    predictions_cat.extend(pred_cat.cpu().detach().numpy())
                    predictions_llm.extend(pred_llm.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            test_loss_molclr /= num_data
            test_loss_cat /= num_data
            test_loss_llm /= num_data

        model.train()

        predictions_molclr = np.array(predictions_molclr)
        predictions_cat = np.array(predictions_cat)
        predictions_llm = np.array(predictions_llm)
        labels = np.array(labels)

        if self.config['dataset']['task'] == 'regression':
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                self.mae_molclr = mean_absolute_error(labels, predictions_molclr)
                self.mae_cat = mean_absolute_error(labels, predictions_cat)
                self.mae_llm = mean_absolute_error(labels, predictions_llm)
                print('Test loss molclr:', test_loss_molclr, 'Test MAE molclr:', self.mae_molclr,
                      '\nTest loss cat:', test_loss_cat, 'Test MAE cat:', self.mae_cat,
                      '\nTest loss llm:', test_loss_llm, 'Test MAE llm:', self.mae_llm)
            else:
                self.rmse_molclr = mean_squared_error(labels, predictions_molclr, squared=False)
                self.rmse_cat = mean_squared_error(labels, predictions_cat, squared=False)
                self.rmse_llm = mean_squared_error(labels, predictions_llm, squared=False)
                print('Test loss molclr:', test_loss_molclr, 'Test RMSE molclr:', self.rmse_molclr,
                      '\nTest loss cat:', test_loss_cat, 'Test RMSE cat:', self.rmse_cat,
                      '\nTest loss llm:', test_loss_llm, 'Test RMSE llm:', self.rmse_llm)

        elif self.config['dataset']['task'] == 'classification':
            self.roc_auc_molclr = roc_auc_score(labels, predictions_molclr[:, 1])
            self.roc_auc_cat = roc_auc_score(labels, predictions_cat[:, 1])
            self.roc_auc_llm = roc_auc_score(labels, predictions_llm[:, 1])
            print('Test loss molclr:', test_loss_molclr, 'Test ROC AUC molclr:', self.roc_auc_molclr,
                  '\nTest loss cat:', test_loss_cat, 'Test ROC AUC cat:', self.roc_auc_cat,
                  '\nTest loss llm:', test_loss_llm, 'Test ROC AUC llm:', self.roc_auc_llm)


def extract_features_and_labels(data_loader):
    features = []
    labels = []
    for batch in data_loader:
        # Access the llm4sd_x and y directly from the batch
        llm4sd_x_batch = batch.llm4sd_x.numpy()
        y_batch = batch.y.numpy()

        for llm4sd_x, y in zip(llm4sd_x_batch, y_batch):
            features.append(llm4sd_x)
            labels.append(y)

    features = np.array(features)
    labels = np.concatenate(labels, axis=0)
    return features, labels


def main(config):
    dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])
    train_l, valid_l, test_l = dataset.get_data_loaders()
    example_data = next(iter(train_l))
    config['model']['llm4sd_x_dim'] = example_data.llm4sd_x.shape[1]

    print("Get MolCLR and combination score ...")
    fine_tune = FineTune(dataset, config)
    fine_tune.train()

    print("Get LLM4SD score ...")
    # llm4sd_score = llm4sd_evaluation(train_l, valid_l, test_l, config['task_name'], config['dataset']["subtask"])
    # llm4sd_score = 0

    if config['dataset']['task'] == 'classification':
        return [fine_tune.roc_auc_llm, fine_tune.roc_auc_molclr, fine_tune.roc_auc_cat]
    if config['dataset']['task'] == 'regression':
        if config['task_name'] in ['qm7', 'qm8', 'qm9']:
            return [fine_tune.mae_llm, fine_tune.mae_molclr, fine_tune.mae_cat]
        else:
            return [fine_tune.rmse_llm, fine_tune.rmse_molclr, fine_tune.rmse_cat]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, default='scaffold_datasets', help="load train/valid/test dataset")
    parser.add_argument('--dataset', type=str, default='freesolv', help='dataset name')
    parser.add_argument('--subtask', type=str, default='', help='subtask of tox21/sider/qm9 dataset')
    parser.add_argument('--model', type=str, default='galactica-6.7b', help='LLM model')
    parser.add_argument('--knowledge_type', type=str, default='all', help='synthesize/inference/all')
    parser.add_argument('--num_samples', type=int, default=30, help='number of sample lists (30/50) for inference')
    parser.add_argument('--feat_type', type=str, default='concat', help='concat or plus')
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

    headers = ['target', 'llm4sd_score', 'MolCLR', config['feat_type'], 'knowledge_type']

    results_list = []
    for target in target_list:
        config['dataset']['target'] = target
        result = main(config)

        if args.knowledge_type == "synthesize":
            results_list.append([target, result[0], result[1], result[2], args.knowledge_type])
        else:
            results_list.append([target, result[0], result[1], result[2], args.knowledge_type, args.num_samples])
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