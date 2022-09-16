from abc import ABC, abstractmethod

import pandas as pd

from placenta.eval.eval import plot_confusion_matrix


class _Appender(ABC):
    @abstractmethod
    def log_accuracy(self, split_name, epoch_num, accuracy):
        pass

    @abstractmethod
    def log_loss(self, split_name, epoch_num, loss):
        pass

    @abstractmethod
    def log_confusion_matrix(self, cm, dataset_name, save_dir):
        pass


class Console(_Appender):
    def log_accuracy(self, split_name, epoch_num, accuracy):
        print(f"{split_name} accuracy: {accuracy}")

    def log_loss(self, split_name, epoch_num, loss):
        print(f"{split_name} loss: {loss}")

    def log_confusion_matrix(self, cm, dataset_name, save_dir):
        print(f"{dataset_name} confusion matrix:")
        print(cm)


class File(_Appender):
    def __init__(self, dataset_names, metrics):
        self.train_stats = self._setup_train_stats(dataset_names, metrics)

    def log_accuracy(self, split_name, epoch_num, accuracy):
        self._add_to_train_stats(epoch_num, split_name, "accuracy", accuracy)

    def log_loss(self, split_name, epoch_num, loss):
        self._add_to_train_stats(epoch_num, split_name, "loss", loss)
        
    def log_confusion_matrix(self, cm, dataset_name, save_dir):
        plot_confusion_matrix(cm, dataset_name, save_dir)

    def _setup_train_stats(self, dataset_names, metrics):
        columns = []
        for name in dataset_names:
            for metric in metrics:
                col = f"{name}_{metric}"
                columns.append(col)
        return pd.DataFrame(columns=columns)

    def _add_to_train_stats(self, epoch_num, dataset_name, metric_name, metric):
        column_name = f"{dataset_name}_{metric_name}"
        if not epoch_num in self.train_stats.index:
            row = pd.Series([metric], index=[column_name])
            self.train_stats = self.train_stats.append(row, ignore_index=True)
        else:
            self.train_stats.loc[epoch_num][column_name] = metric
