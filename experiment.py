import os
from datetime import datetime, timedelta
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np

from model_factory import get_model
from criteria import dice_coeff, get_criterion
from dataset_factory import get_datasets, get_test_dataset
from file_utils import *
import matplotlib.pyplot as plt
from constants import ROOT_STATS_DIR
from test_helper import save_test_results
from transforms import PadToSquare


class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./config/', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        # Load Datasets
        self.name = config_data['experiment_name']
        self.experiment_dir = os.path.join(ROOT_STATS_DIR, self.name)

        ds_train, ds_val = get_datasets(config_data)
        self.train_loader = DataLoader(ds_train, batch_size=config_data['experiment']['batch_size_train'], shuffle=True,
                                       num_workers=config_data['experiment']['num_workers'], pin_memory=True)
        self.val_loader = DataLoader(ds_val, batch_size=config_data['experiment']['batch_size_val'], shuffle=True,
                                     num_workers=config_data['experiment']['num_workers'], pin_memory=True)

        ds_test = get_test_dataset(config_data)
        self.test_loader = DataLoader(ds_test, batch_size=1, num_workers=config_data['experiment']['num_workers'],
                                      pin_memory=True)

        # Setup Experiment Stats
        self.epochs = config_data['experiment']['num_epochs']
        self.current_epoch = 0
        self.training_losses = []
        self.val_losses = []
        self.val_dices = []
        self.ed_dices = []
        self.es_dices = []

        # Init Model and Criterion
        self.criterion = get_criterion(config_data)
        self.model = get_model(config_data)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config_data['experiment']['learning_rate'])
        self.init_model()
        self.ensemble = config_data['model']['ensemble']

        # Load Experiment Data if available
        self.load_experiment()
        self.log(str(config_data))

    def load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.experiment_dir):
            self.training_losses = read_file_in_dir(self.experiment_dir, 'training_losses.txt')
            self.val_losses = read_file_in_dir(self.experiment_dir, 'val_losses.txt')
            self.val_dices = read_file_in_dir(self.experiment_dir, 'val_dices.txt')
            self.ed_dices = read_file_in_dir(self.experiment_dir, 'ed_dices.txt')
            self.es_dices = read_file_in_dir(self.experiment_dir, 'es_dices.txt')

            if len(self.ed_dices) == 0:  # Backward Compatibility
                self.ed_dices = [0] * len(self.training_losses)
                self.es_dices = [0] * len(self.training_losses)

            self.current_epoch = len(self.training_losses)
            state_dict = torch.load(os.path.join(self.experiment_dir, 'latest_model.pt'))
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.experiment_dir)
            os.makedirs(os.path.join(self.experiment_dir, 'models'))

    def load_model_at_epoch(self, epoch_num):
        epoch_model_path = os.path.join(self.experiment_dir, 'models', 'model_{}.pt'.format(epoch_num))
        state_dict = torch.load(epoch_model_path)
        self.model.load_state_dict(state_dict)

    def init_model(self):
        if torch.cuda.is_available():
            self.model = self.model.cuda().float()
        else:
            self.model = self.model.double()
        # self.model = torch.nn.DataParallel(self.model)

    @staticmethod
    def __smooth(array, smooth_factor=2):
        out = []
        for i in range(smooth_factor):
            out.append(array[i])
        for i in range(smooth_factor, len(array) - smooth_factor):
            out.append(sum(array[i - smooth_factor: i + smooth_factor + 1]) / (2 * smooth_factor + 1))
        return out

    def get_perf_stats(self):
        results = {}
        results['current'] = self.__get_perf()

        smooth_losses = self.__smooth(self.val_losses)
        best_epoch = np.argmin(smooth_losses)
        print("Best Loss Epoch", best_epoch)
        self.load_model_at_epoch(best_epoch)
        results['best_loss'] = self.__get_perf()
        results['best_loss_epoch'] = best_epoch

        smooth_dices = self.__smooth(self.val_dices)
        best_epoch = np.argmax(smooth_dices)
        print("Best Dice Epoch", best_epoch)
        self.load_model_at_epoch(best_epoch)
        results['best_dice'] = self.__get_perf()
        results['best_dice_epoch'] = best_epoch

        self.load_model_at_epoch(len(self.val_losses) - 1)
        print(results)
        return results

    def __get_perf(self):
        l1, d1, ed1, es1 = self.val()
        l2, d2, ed2, es2 = self.val()
        l3, d3, ed3, es3 = self.val()
        return ((l1 + l2 + l3) / 3, (d1 + d2 + d3) / 3, (ed1 + ed2 + ed3) / 3, (es1 + es2 + es3) / 3)

    def run(self):
        start_epoch = self.current_epoch
        for epoch in range(start_epoch, self.epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.current_epoch = epoch
            train_loss = self.train()
            val_loss, val_dice, ed_dice, es_dice = self.val()
            self.record_stats(train_loss, val_loss, val_dice, ed_dice, es_dice)
            self.log_epoch_stats(start_time)
            self.save_model()
            self.plot_sample_outputs()

    def train(self):
        self.model.train()
        train_loss_epoch = []
        for i, data in enumerate(self.train_loader):
            inputs = data[0].cuda().float() if torch.cuda.is_available() else data[0].double()
            labels = data[1].cuda().float() if torch.cuda.is_available() else data[1].double()
            frame_types = data[2]

            self.optimizer.zero_grad()
            outputs = self.model.forward(inputs, frame_types) if self.ensemble else self.model.forward(inputs)
            loss = self.criterion(outputs.squeeze(), labels.squeeze())
            loss.backward()
            self.optimizer.step()
            train_loss_epoch.append(loss.item())

            status_str = "Epoch: {}, Train, Batch {}/{}. Loss {}".format(self.current_epoch + 1, i + 1,
                                                                         len(self.train_loader),
                                                                         loss.item())
            self.log(status_str)

        return np.mean(train_loss_epoch)

    def val(self):
        self.model.eval()
        val_loss_epoch = []
        val_dice_epoch = []
        ed_dice_epoch = []
        es_dice_epoch = []
        ed_slices = 0
        es_slices = 0
        for i, data in enumerate(self.val_loader):
            inputs = data[0].cuda().float() if torch.cuda.is_available() else data[0].double()
            labels = data[1].cuda().float() if torch.cuda.is_available() else data[1].double()
            frame_types = data[2]

            with torch.no_grad():
                outputs = self.model.forward(inputs, frame_types) if self.ensemble else self.model.forward(inputs)
                loss = self.criterion(outputs.squeeze(), labels.squeeze())
                dice = dice_coeff(outputs, labels)
                ed_dice = dice_coeff(outputs[frame_types], labels[frame_types])
                es_dice = dice_coeff(outputs[~frame_types], labels[~frame_types])
            val_loss_epoch.append(loss.item())
            val_dice_epoch.append(dice.item())
            ed_dice_epoch.append(ed_dice.item() * sum(frame_types))
            es_dice_epoch.append(es_dice.item() * sum(~frame_types))
            ed_slices += sum(frame_types)
            es_slices += sum(~frame_types)

            status_str = "Epoch: {}, Val, Batch {}/{}. Loss {}".format(self.current_epoch + 1, i + 1,
                                                                       len(self.val_loader),
                                                                       loss.item())
            self.log(status_str)

        ed_dice, es_dice = sum(ed_dice_epoch) / ed_slices, sum(es_dice_epoch) / es_slices
        return np.mean(val_loss_epoch), np.mean(val_dice_epoch), ed_dice.item(), es_dice.item()

    def plot_sample_outputs(self):
        self.model.eval()

        val_iter = iter(self.val_loader)
        data = next(val_iter)
        inputs = data[0].cuda().float() if torch.cuda.is_available() else data[0].double()
        labels = data[1].numpy()
        frame_types = data[2]

        with torch.no_grad():
            outputs = self.model.forward(inputs, frame_types) if self.ensemble else self.model.forward(inputs)
            outputs = outputs.squeeze()

        if inputs.dim() == 5:
            inputs = inputs[:, -1, :]

        inputs = inputs.cpu().numpy()
        predictions = (torch.nn.Sigmoid()(outputs) > 0.5).int().cpu().numpy()
        num_samples = labels.shape[0]

        fig, axes = plt.subplots(nrows=num_samples, ncols=3, figsize=(30, 40))
        for i in range(num_samples):
            axes[i][0].imshow(inputs[i, :, :].squeeze())
            axes[i][0].set_title('Input')
            axes[i][0].axis('off')
            axes[i][1].imshow(labels[i, :, :].squeeze())
            axes[i][1].set_title('Actual')
            axes[i][1].axis('off')
            axes[i][2].imshow(predictions[i, :, :].squeeze())
            axes[i][2].set_title('Prediction')
            axes[i][2].axis('off')
        fig.tight_layout()
        out_dir = os.path.join(self.experiment_dir, 'results')
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, "{}.png".format(self.current_epoch)))
        plt.show()

    def test(self):
        self.model.eval()
        results = {}
        for i, data in enumerate(self.test_loader):
            inputs = data[0].cuda().float() if torch.cuda.is_available() else data[0].double()
            file_path = data[1][0].strip()
            frame_types = data[2]
            print('Evaluating ', file_path)
            with torch.no_grad():
                shape = inputs.shape[-2:]
                dim = inputs.dim()
                pad = PadToSquare(axes=(dim - 2, dim - 1))
                inputs = pad(inputs)
                output = self.model.forward(inputs, frame_types) if self.ensemble else self.model.forward(inputs)
                prediction = (torch.nn.Sigmoid()(output) > 0.5).int().cpu().numpy().squeeze()
                results[file_path] = {'data': prediction, 'shape': shape}
        save_test_results(self.experiment_dir, results)

    def save_model(self):
        epoch_model_path = os.path.join(self.experiment_dir, 'models', 'model_{}.pt'.format(self.current_epoch))
        root_model_path = os.path.join(self.experiment_dir, 'latest_model.pt')

        if isinstance(self.model, torch.nn.DataParallel):
            model_dict = self.model.module.state_dict()
        else:
            model_dict = self.model.state_dict()

        state_dict = {'model': model_dict, 'optimizer': self.optimizer.state_dict()}
        torch.save(self.model.state_dict(), epoch_model_path)
        torch.save(state_dict, root_model_path)

    def record_stats(self, train_loss, val_loss, val_dice, ed_dice, es_dice):
        self.training_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_dices.append(val_dice)
        self.ed_dices.append(ed_dice)
        self.es_dices.append(es_dice)

        self.plot_stats()

        write_to_file_in_dir(self.experiment_dir, 'training_losses.txt', self.training_losses)
        write_to_file_in_dir(self.experiment_dir, 'val_losses.txt', self.val_losses)
        write_to_file_in_dir(self.experiment_dir, 'val_dices.txt', self.val_dices)
        write_to_file_in_dir(self.experiment_dir, 'ed_dices.txt', self.ed_dices)
        write_to_file_in_dir(self.experiment_dir, 'es_dices.txt', self.es_dices)

    def log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.experiment_dir, file_name, log_str)

    def log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.epochs - self.current_epoch - 1)
        train_loss = self.training_losses[self.current_epoch]
        val_loss = self.val_losses[self.current_epoch]
        val_dice = self.val_dices[self.current_epoch]
        ed_dice = self.es_dices[-1]
        es_dice = self.es_dices[-1]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Val Dice: {}, ED: {}, ES: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.current_epoch + 1, train_loss, val_loss, val_dice, ed_dice, es_dice,
                                         str(time_elapsed),
                                         str(time_to_completion))
        self.log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.training_losses)
        x_axis = np.arange(1, e + 1, 1)
        fig, (a1, a2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        a1.plot(x_axis, self.training_losses, label="Training Loss")
        a1.plot(x_axis, self.val_losses, label="Validation Loss")
        a1.set(xlabel='Epochs', ylabel='Loss')
        a1.legend(loc='best')
        a1.set_title(self.name + " Loss Plot")
        a2.plot(x_axis, self.val_dices, label="Total Val Dice")
        a2.plot(x_axis, self.ed_dices, label="ED Val Dice")
        a2.plot(x_axis, self.es_dices, label="ES Val Dice")
        a2.set(xlabel='Epochs', ylabel='Dice Score')
        a2.legend(loc='best')
        a2.set_title(self.name + " Dice Plot")
        fig.savefig(os.path.join(self.experiment_dir, "stat_plot.png"))
        plt.show()
