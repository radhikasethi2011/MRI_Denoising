import os
import sys
import core.logger as Logger
import logging
import argparse
import torch
from collections import OrderedDict
from data import create_dataset, create_dataloader
from model import create_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args, stage=1)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')

    # GPU setup
    logger.info(f'PyTorch version: {torch.__version__}')
    logger.info(f'CUDA version: {torch.version.cuda}')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, opt.gpu_ids))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f'Using device: {device}')

    # dataset
    train_set = create_dataset(opt.dataset.train)
    train_loader = create_dataloader(train_set, opt.dataset.train, pin_memory=True)
    val_set = create_dataset(opt.dataset.val)
    val_loader = create_dataloader(val_set, opt.dataset.val, pin_memory=True)

    logger.info(f'Loaded training dataset of size: {len(train_loader.dataset)}')
    logger.info(f'Loaded validation dataset of size: {len(val_loader.dataset)}')

    # model
    model = create_model(opt.model)
    model.setup(opt)
    model = model.to(device)

    # training
    current_step = 0
    start_epoch = 0
    total_iters = len(train_loader)
    
    for epoch in range(start_epoch, opt.train.n_iter):
        for i, train_data in enumerate(train_loader):
            current_step += 1
            train_data = {key: val.to(device) if isinstance(val, torch.Tensor) else val for key, val in train_data.items()}
            model.feed_data(train_data)
            model.optimize_parameters()
            
            if current_step % opt.train.print_freq == 0:
                logger.info(f'<epoch: {epoch}, iter: {current_step}, total_iters: {total_iters}> l_pix: {model.get_current_log()}')

            if current_step % opt.train.val_freq == 0:
                for _, val_data in enumerate(val_loader):
                    val_data = {key: val.to(device) if isinstance(val, torch.Tensor) else val for key, val in val_data.items()}
                    model.feed_data(val_data)
                    model.test()
                    visuals = model.get_current_visuals()
                    logger.info(f'Validation output: {visuals}')

            if current_step % opt.train.save_checkpoint_freq == 0:
                logger.info('Saving models and training states.')
                model.save('latest')
                model.save_training_state(epoch, current_step)

    logger.info('Training completed.')


