import collections

from happy.logger.appenders import Console, File, Visdom


class Logger:
    def __init__(self, dataset_names, metrics, file=True):
        self.loss_hist = collections.deque(maxlen=500)
        self.appenders = self._get_appenders(file, dataset_names, metrics)

    def log_accuracy(self, split_name, epoch_num, accuracy):
        for a in self.appenders:
            self.appenders[a].log_accuracy(split_name, epoch_num, round(accuracy, 4))

    def log_loss(self, split_name, epoch_num, loss):
        for a in self.appenders:
            self.appenders[a].log_loss(split_name, epoch_num, round(loss, 4))

    def log_confusion_matrix(self, cm, dataset_name, save_dir):
        for a in self.appenders:
            self.appenders[a].log_confusion_matrix(cm, dataset_name, save_dir)

    def to_csv(self, save_path):
        file_appender = self.appenders["file"]
        file_appender.train_stats.to_csv(save_path)

    def _get_appenders(self, file, dataset_names, metrics):
        appenders = {"console": Console()}
        if file:
            appenders["file"] = File(dataset_names, metrics)
        return appenders
