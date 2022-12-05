import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
from logger import logger
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

LOGGER_FILE = "logs/"+ world.model_name +"_bpr.csv"
utils.create_subfolders_if_not(LOGGER_FILE)

pretrain_modelname = "mf"
weight_file = utils.getFileName_pre(pretrain_modelname) 
if world.config['pretrain']:
    try:
        print('Got here')
        pre_model = register.MODELS[pretrain_modelname](world.config, dataset)
        pre_model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        world.config['user_emb'] = pre_model.embedding_user.weight.data.numpy()
        world.config['item_emb'] = pre_model.embedding_item.weight.data.numpy()
    except FileNotFoundError:
        print(f"{weight_file} does not exist, start from beginning")


utils.log_training_events(["Epoch", "Loss", "Recall", "Precision", "NDCG"], LOGGER_FILE, reset=True)
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch %1 == 0:
            cprint("[TEST]")
            results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
        output_information, aver_loss = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        utils.log_training_events([epoch, aver_loss, results['recall'].item(), results['precision'].item(), results['ndcg'].item()], LOGGER_FILE)
        torch.save(Recmodel.state_dict(), weight_file)
finally:
    if world.tensorboard:
        w.close()
