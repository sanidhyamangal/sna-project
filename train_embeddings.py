"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
from argparse import ArgumentParser  # for argument parser

import numpy as np  # for np based ops
import torch  # for torch based ops
from sklearn.metrics import precision_score, recall_score
from torch.nn import MSELoss  # for mse loss computation
from torch.optim import Adam  # for adam ops

from dataloader.utils import \
    create_subfolders_if_not  # for creation of subfolder if not exists
from logger import logger
from matrix_factorization.dataclass import MF_DataReader  # for data loader
from matrix_factorization.models import MF  # for mf ops
from utils import DEVICE, log_training_events


def train_embeddings(dataset: str,
                     batch_size: int,
                     latent_space: int,
                     logger_file: str,
                     path_to_save_model: int,
                     delimeter: str,
                     file_ext: str,
                     epochs: int = 100):
    # define training params
    DATASET = dataset
    EPOCHS = epochs
    LOGGER_FILE = logger_file
    create_subfolders_if_not(LOGGER_FILE)
    batch_size = batch_size
    MODEL_SAVE = path_to_save_model
    create_subfolders_if_not(MODEL_SAVE)
    log_training_events(["Epoch", "Loss", "Accuracy", "Recall", "Precision"],
                        LOGGER_FILE,
                        reset=True)

    # data loader
    datareader = MF_DataReader(train_set=DATASET + f"/train.{file_ext}",
                               test_set=DATASET + f"/test.{file_ext}",
                               batch_size=batch_size,
                               delimeter=delimeter)

    # model and optimizer
    model = MF(datareader.n_user, datareader.m_item, latent_space).to(DEVICE())
    optimizer = Adam(model.parameters())
    criterion = MSELoss()

    EPOCH_LOSS, EPOCH_ACCURACY = [], []

    # training loop
    for i in range(EPOCHS):
        _loss, _accuracy = [], []

        model.train()
        for batch in datareader.iterator():
            optimizer.zero_grad()
            pred = model(
                torch.LongTensor(batch['user']).to(device=DEVICE()),
                torch.LongTensor(batch['items']).to(device=DEVICE()))
            loss = criterion(
                torch.FloatTensor(batch['labels']).to(device=DEVICE()), pred)

            _loss.append(loss.detach().cpu().item())
            loss.backward()
            optimizer.step()

        EPOCH_LOSS.append(np.mean(_loss))

        model.eval()
        true_labels, actual_preds = [], []
        for test_batch in datareader.iterator(True):
            test_pred = model(
                torch.LongTensor(test_batch['user']).to(device=DEVICE()),
                torch.LongTensor(test_batch['items']).to(device=DEVICE()))
            total = pred.shape[0]
            # compute the f1 score for the accuracy metric.
            test_pred[test_pred >= 0.5] = 1
            test_pred[test_pred < 0.5] = 0
            correct = (test_batch['labels'].flatten() == test_pred.detach().
                       cpu().numpy().flatten()).sum().item()
            accuracy = correct / total
            true_labels.extend(test_batch['labels'].flatten())
            actual_preds.extend(test_pred.detach().cpu().numpy().flatten())
            _accuracy.append(accuracy)

        EPOCH_ACCURACY.append(np.mean(_accuracy))

        logger.info(
            f"Epoch:{i}, Loss:{EPOCH_LOSS[-1]}, Accuracy:{EPOCH_ACCURACY[-1]}, Recall: {recall_score(true_labels, actual_preds)}"
        )
        log_training_events([
            i, EPOCH_LOSS[-1], EPOCH_ACCURACY[-1],
            recall_score(true_labels, actual_preds),
            precision_score(true_labels, actual_preds)
        ], LOGGER_FILE)

    torch.save(model.state_dict(), MODEL_SAVE)


if __name__ == "__main__":
    # argparser for training the model via this script
    argparser = ArgumentParser(
        "Script to train Matrix Factorization pre-training")
    argparser.add_argument("--dataset",
                           help="Dataset",
                           default="processed_data/ebid")
    argparser.add_argument("--batch_size",
                           help="Batch Size to train the model",
                           default=64,
                           type=int)
    argparser.add_argument("--latent_space",
                           help="Latent Space of MF model",
                           default=64,
                           type=int)
    argparser.add_argument("--epochs",
                           help="Epochs to run the code for",
                           default=100,
                           type=int)
    argparser.add_argument("--logger_file",
                           help="Path to save logger file",
                           default="logs/ebid.csv")
    argparser.add_argument("--path_to_model",
                           help="Path to save model file",
                           default="code/code/checkpoints/pmf-ebid-64.pth.tar")
    argparser.add_argument("--extension",
                           help="file extension, default csv",
                           default="csv")
    argparser.add_argument("--delimeter",
                           help="delimeter for seperating files",
                           default=" ",
                           required=False)

    args = argparser.parse_args()

    train_embeddings(args.dataset, args.batch_size, args.latent_space,
                     args.logger_file, args.path_to_model, args.delimeter,
                     args.extension, args.epochs)
