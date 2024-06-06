import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
import os
import numpy as np

print(torch.version.cuda)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    stage2_file = opt['stage2_file']

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase, stage2_file=stage2_file)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase, stage2_file=stage2_file)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Validation
    logger.info('Begin Model Evaluation.')
    avg_psnr = 0.0
    avg_ssim = 0.0
    idx = 0
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    for _,  val_data in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals()
        
        # Check if 'Y' is in visuals
        if 'Y' not in visuals:
            logger.error(f"Key 'Y' not found in visuals dictionary: {visuals.keys()}")
            continue

        denoised_img = Metrics.tensor2img(visuals['denoised'])  # uint8
        input_img = Metrics.tensor2img(visuals['X'])  # uint8
        target_img = Metrics.tensor2img(visuals['Y'])  # uint8

        Metrics.save_img(
            denoised_img[:,:], '{}/{}_{}_denoised.png'.format(result_path, current_step, idx))
        Metrics.save_img(
            input_img[:,:], '{}/{}_{}_input.png'.format(result_path, current_step, idx))
        Metrics.save_img(
            target_img[:,:], '{}/{}_{}_target.png'.format(result_path, current_step, idx))

    logger.info('End of evaluation.')

if __name__ == "__main__":
    main()
