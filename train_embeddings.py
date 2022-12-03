"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
import numpy as np  # for np based ops
import torch  # for torch based ops
from torch.nn import MSELoss  # for mse loss computation
from torch.optim import Adam  # for adam ops

from dataloader.utils import \
    create_subfolders_if_not  # for creation of subfolder if not exists
from logger import logger
from matrix_factorization.dataclass import MF_DataReader  # for data loader
from matrix_factorization.models import MF  # for mf ops
from utils import DEVICE, log_training_events

DATASET = "processed_data/ebid"
EPOCHS = 1_00
LOGGER_FILE = "logs/mf.csv"
create_subfolders_if_not(LOGGER_FILE)
batch_size = 64
MODEL_SAVE = "trained_model/mf.pt"
create_subfolders_if_not(MODEL_SAVE)
log_training_events(["Epoch", "Loss", "Accuracy"], LOGGER_FILE, reset=True)

datareader = MF_DataReader(train_set=DATASET + "/user_item_train.csv",
                           test_set=DATASET + "/user_item_test.csv",
                           batch_size=batch_size)
model = MF(datareader.n_user, datareader.m_item, 64).to(DEVICE())
optimizer = Adam(model.parameters())
criterion = MSELoss()

EPOCH_LOSS, EPOCH_ACCURACY = [], []

for i in range(EPOCHS):
    _loss, _accuracy = [], []

    model.train()
    for batch in datareader.iterator():
        optimizer.zero_grad()
        pred = model(torch.LongTensor(batch['user'], device=DEVICE()),
                     torch.LongTensor(batch['items'], device=DEVICE()))
        loss = criterion(torch.FloatTensor(batch['labels'], device=DEVICE()),
                         pred)

        _loss.append(loss.detach().cpu().item())
        loss.backward()
        optimizer.step()

    EPOCH_LOSS.append(np.mean(_loss))

    model.eval()
    for test_batch in datareader.iterator(True):
        test_pred = model(
            torch.LongTensor(test_batch['user'], device=DEVICE()),
            torch.LongTensor(test_batch['items'], device=DEVICE()))
        total = pred.shape[0]
        # compute the f1 score for the accuracy metric.
        test_pred[test_pred >= 0.5] = 1
        test_pred[test_pred < 0.5] = 0
        correct = (test_batch['labels'].flatten() ==
                   test_pred.detach().cpu().numpy().flatten()).sum().item()
        accuracy = correct / total
        _accuracy.append(accuracy)

    EPOCH_ACCURACY.append(np.mean(_accuracy))

    logger.info(
        f"Epoch:{i}, Loss:{EPOCH_LOSS[-1]}, Accuracy:{EPOCH_ACCURACY[-1]}")
    log_training_events([i, EPOCH_LOSS[-1], EPOCH_ACCURACY[-1]], LOGGER_FILE)

torch.save(model, MODEL_SAVE)
