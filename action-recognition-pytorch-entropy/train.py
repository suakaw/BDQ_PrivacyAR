import os
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
import shutil
import time
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.multiprocessing as mp
from torch.optim import lr_scheduler
import tensorboard_logger
import torch.nn.functional as F
import csv

from models._internally_replaced_utils import load_state_dict_from_url

from models import build_model
from utils.utils import (train, validate,train1, validate1, build_dataflow, get_augmentor,
                         save_checkpoint)
from utils.video_dataset import VideoDataSet
from utils.dataset_config import get_dataset_config
from opts import arg_parser


import torch
from  torch.nn.modules.loss import _Loss


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')


        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

class EntropyLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(EntropyLoss, self).__init__(size_average, reduce, reduction)

    # input is probability distribution of output classes
    def forward(self, input):
        if (input < 0).any() or (input > 1).any():
            raise Exception('Entropy Loss takes probabilities 0<=input<=1')

        input = input + 1e-16  # for numerical stability while taking log
        H = -torch.mean(torch.sum(input * torch.log(input), dim=1))

        return H


def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function

        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    cudnn.benchmark = args.cudnn_benchmark
    args.gpu = gpu

    num_classes, train_list_name, val_list_name, test_list_name, filename_seperator, image_tmpl, filter_video, label_file = get_dataset_config(
        args.dataset)
    args.num_classes = num_classes

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.modality == 'rgb':
        args.input_channels = 3
    elif args.modality == 'flow':
        args.input_channels = 2 * 5

    model, arch_name = build_model(args)

    arch_name = 'privacy_action_recognition'

    model_degrad= model[0]
    model_target= model[1]
    model_budget= model[2]

    model_target1= model[3]
    model_budget1= model[4]


    '''
    mean = model.mean(args.modality)
    std = model.std(args.modality)

    # overwrite mean and std if they are presented in command
    if args.mean is not None:
        if args.modality == 'rgb':
            if len(args.mean) != 3:
                raise ValueError("When training with rgb, dim of mean must be three.")
        elif args.modality == 'flow':
            if len(args.mean) != 1:
                raise ValueError("When training with flow, dim of mean must be three.")
        mean = args.mean

    if args.std is not None:
        if args.modality == 'rgb':
            if len(args.std) != 3:
                raise ValueError("When training with rgb, dim of std must be three.")
        elif args.modality == 'flow':
            if len(args.std) != 1:
                raise ValueError("When training with flow, dim of std must be three.")
        std = args.std
    '''
    model_degrad = model_degrad.cuda(args.gpu)
    model_target = model_target.cuda(args.gpu)
    model_budget = model_budget.cuda(args.gpu)
    model_target1 = model_target1.cuda(args.gpu)
    model_budget1 = model_budget1.cuda(args.gpu)

    '''
    model_degrad.eval()
    model_target.eval()
    model_budget.eval()
    model_target1.eval()
    model_budget1.eval()
    '''
    if args.show_model:
        if args.rank == 0:
            print(model_degrad)
            print(model_target)
            print(model_budget)
            print(model_target1)
            print(model_budget1)

        return 0
    
    
    if args.pretrained is not None:
        if args.rank == 0:
            print("=> using pre-trained model '{}'".format(arch_name))
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()
    else:
        if args.rank == 0:
            print("=> creating model '{}'".format(arch_name))


    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            # the batch size should be divided by number of nodes as well
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int(args.workers / ngpus_per_node)

            if args.sync_bn:
                process_group = torch.distributed.new_group(list(range(args.world_size)))
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)

            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model_degrad = model_degrad.cuda(args.gpu)
        model_target = model_target.cuda(args.gpu)
        model_budget = model_budget.cuda(args.gpu)
        model_target1 = model_target1.cuda(args.gpu)
        model_budget1 = model_budget1.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        # assign rank to 0
        model_degrad = torch.nn.DataParallel(model_degrad).cuda()
        model_target = torch.nn.DataParallel(model_target).cuda()
        model_budget = torch.nn.DataParallel(model_budget).cuda()
        model_target1 = torch.nn.DataParallel(model_target1).cuda()
        model_budget1 = torch.nn.DataParallel(model_budget1).cuda()

        args.rank = 0



    # define loss function (criterion) and optimizer
    train_criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    val_criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    train_entropy_criterion = EntropyLoss().cuda(args.gpu)
    val_entropy_criterion = EntropyLoss().cuda(args.gpu)
    reconst_criterion = nn.MSELoss().cuda(args.gpu)

    # Data loading code
    val_list = os.path.join(args.datadir, val_list_name)

    norm_value= 255

    val_augmentor = get_augmentor(False, args.input_size, scale_range=args.scale_range, mean=[110.63666788 / norm_value, 103.16065604 / norm_value, 96.29023126 / norm_value],
                                  std=[38.7568578 / norm_value, 37.88248729 / norm_value, 40.02898126 / norm_value], disable_scaleup=args.disable_scaleup,
                                  threed_data=args.threed_data,
                                  is_flow=True if args.modality == 'flow' else False,
                                  version=args.augmentor_ver)

    val_dataset = VideoDataSet(args.datadir, val_list, args.groups, args.frames_per_group,
                               num_clips=args.num_clips,
                               modality=args.modality, image_tmpl=image_tmpl,
                               dense_sampling=args.dense_sampling,
                               transform=val_augmentor, is_train=False, test_mode=False,
                               seperator=filename_seperator, filter_video=filter_video)

    val_loader = build_dataflow(val_dataset, is_train=False, batch_size=args.batch_size,
                                workers=args.workers,
                                is_distributed=args.distributed)

    log_folder = os.path.join(args.logdir, arch_name)
    if args.rank == 0:
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

    if args.evaluate:
        val_top1, val_top5, val_losses, val_speed = validate(val_loader, model, val_criterion,
                                                             gpu_id=args.gpu)
        if args.rank == 0:
            logfile = open(os.path.join(log_folder, 'evaluate_log.log'), 'a')
            print(
                'Val@{}: \tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\tSpeed: {:.2f} ms/batch'.format(
                    args.input_size, val_losses, val_top1, val_top5, val_speed * 1000.0),
                flush=True)
            print(
                'Val@{}: \tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\tSpeed: {:.2f} ms/batch'.format(
                    args.input_size, val_losses, val_top1, val_top5, val_speed * 1000.0),
                flush=True,
                file=logfile)
        return

    train_list = os.path.join(args.datadir, train_list_name)

    train_augmentor = get_augmentor(False, args.input_size, scale_range=args.scale_range, mean=[110.63666788 / norm_value, 103.16065604 / norm_value, 96.29023126 / norm_value],
                                  std=[38.7568578 / norm_value, 37.88248729 / norm_value, 40.02898126 / norm_value],
                                    disable_scaleup=args.disable_scaleup,
                                    threed_data=args.threed_data,
                                    is_flow=True if args.modality == 'flow' else False,
                                    version=args.augmentor_ver)

    train_dataset = VideoDataSet(args.datadir, train_list, args.groups, args.frames_per_group,
                                 num_clips=args.num_clips,
                                 modality=args.modality, image_tmpl=image_tmpl,
                                 dense_sampling=args.dense_sampling,
                                 transform=train_augmentor, is_train=True, test_mode=False,
                                 seperator=filename_seperator, filter_video=filter_video)

    train_loader = build_dataflow(train_dataset, is_train=True, batch_size=args.batch_size,
                                  workers=args.workers, is_distributed=args.distributed)



    
    sys.stdout.flush()
    kinetics_path = ''
    checkpoint = torch.load(kinetics_path)
    model_target.load_state_dict(checkpoint['state_dict'], strict= False)

    sys.stdout.flush()
    imagenet_path = ''
    checkpoint = torch.load(imagenet_path)
    model_budget.load_state_dict(checkpoint['state_dict'], strict= False)
    


    total_epochs= 100


    save_dest = 'results/sbu'
    if not os.path.isdir(save_dest):
        os.mkdir(save_dest)
            

    ctr=5
    while ctr<1:
    #for ctr in down:       
        #----------------- START OF Adv TRAINING------------------#
        ctr+=1

        
        train_logger = Logger(save_dest+'/'+'adv_train_'+str(ctr)+'.log',['epoch','prec1_T', 'prec1_B'])
        val_logger = Logger(save_dest+'/'+'adv_val_'+str(ctr)+'.log',['epoch','prec1_T', 'prec1_B'])
        
        params_t = list(model_target.parameters())+list(model_degrad.parameters())
        params_b = model_budget.parameters()
        optimizer_t = torch.optim.SGD(params_t, args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        optimizer_b = torch.optim.SGD(params_b, args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        scheduler_t = lr_scheduler.CosineAnnealingLR(optimizer_t, T_max= total_epochs, eta_min=1e-7, verbose=True)
        scheduler_b = lr_scheduler.CosineAnnealingLR(optimizer_b, T_max= total_epochs, eta_min=1e-7, verbose=True)
        
        for epoch in range(args.start_epoch, total_epochs):
            
            print('\n Running Adv Training Now '+str(ctr))

            trainT_top1, trainT_top5, trainB_top1, trainB_top5, train_losses, train_speed, speed_data_loader, train_steps = train(train_loader, model_degrad, model_target, model_budget, optimizer_t,  
                                                                                                                 optimizer_b, train_criterion, train_entropy_criterion,  epoch + 1, display =  
                                                                                                                 args.print_freq, clip_gradient=args.clip_gradient, gpu_id= args.gpu, rank=args.rank, step='adv')

            train_logger.log({'epoch': epoch,'prec1_T': trainT_top1.item(), 'prec1_B': trainB_top1.item()})

            if args.distributed:
                dist.barrier()

                
            valT_top1, valT_top5,  valB_top1, valB_top5, val_losses, val_speed = validate(val_loader, model_degrad, model_target, model_budget, val_criterion, gpu_id= args.gpu)


            val_logger.log({'epoch': epoch,'prec1_T': valT_top1.item(), 'prec1_B': valB_top1.item()})

            scheduler_t.step()
            scheduler_b.step()

            if args.distributed:
                dist.barrier()


            print('Train: [{:03d}/{:03d}]\tLoss: {:4.4f}\tTopT@1: {:.4f}\tTopB@1: {:.4f}\t'
                      'Speed: {:.2f} ms/batch\tData loading: {:.2f} ms/batch'.format(
                        epoch + 1, total_epochs, train_losses, trainT_top1, trainB_top1, train_speed * 1000.0,
                        speed_data_loader * 1000.0), flush=True)
               
            print('Val  : [{:03d}/{:03d}]\tLoss: {:4.4f}\tTopT@1: {:.4f}\tTopB@1: {:.4f}\t'
                      'Speed: {:.2f} ms/batch'.format(epoch + 1, total_epochs, val_losses, valT_top1,
                                                      valB_top1, val_speed * 1000.0),flush=True)




            is_best = valT_top1
            best_top1 = valT_top1


            save_dict = {'net': model_degrad,
                         'epoch': epoch,
                         'state_dict': model_degrad.state_dict(),
                         'acc': best_top1,
                         'optimizer': optimizer_t.state_dict(),
                         'scheduler': scheduler_t.state_dict()
                        }

              
            torch.save(save_dict, save_dest+'/'+ 'model_degrad_'+str(ctr)+'.ckpt')
            
              
            save_dict = {'net': model_target,
                         'epoch': epoch,
                         'state_dict': model_target.state_dict(),
                         'acc': best_top1,
                         'optimizer': optimizer_t.state_dict(),
                         'scheduler': scheduler_t.state_dict()
                        }

              
            torch.save(save_dict, save_dest+'/'+ 'model_target_'+str(ctr)+'.ckpt')

              
            save_dict = {'net': model_budget,
                         'epoch': epoch,
                         'state_dict': model_budget.state_dict(),
                         'acc': best_top1,
                         'optimizer': optimizer_b.state_dict(),
                         'scheduler': scheduler_b.state_dict()
                        }

              
            torch.save(save_dict, save_dest+'/'+ 'model_budget_'+str(ctr)+'.ckpt')
        
       
        
        #----------------- END OF Adv TRAINING------------------#

        
        
        #----------------- START OF TARGET MODEL TRAINING------------------#
        
        train_logger = Logger(save_dest+'/'+'model_target_train_'+str(ctr)+'.log',['epoch','prec1_T', 'prec1_B'])
        val_logger = Logger(save_dest+'/'+'model_target_val_'+str(ctr)+'.log',['epoch','prec1_T', 'prec1_B'])

        
        sys.stdout.flush()
        checkpoint = torch.load(kinetics_path)
        model_target1.load_state_dict(checkpoint['state_dict'], strict= False)
        

        
        params = model_target1.parameters()
        optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max= total_epochs, eta_min=1e-7, verbose=True)

        for epoch in range(args.start_epoch, total_epochs):
            
            print('\n Running Target Model Now')
            
            trainT_top1, trainT_top5, trainB_top1, trainB_top5, train_losses, train_speed, speed_data_loader, train_steps = train1(train_loader, model_degrad, model_target1, model_budget1, optimizer,  
                                                                                                                 train_criterion, train_entropy_criterion,  epoch + 1, display =  
                                                                                                                 args.print_freq, clip_gradient=args.clip_gradient, gpu_id= args.gpu, rank=args.rank, step='target', d=ctr)


            train_logger.log({'epoch': epoch,'prec1_T': trainT_top1.item(), 'prec1_B': trainB_top1.item()})
            

            if args.distributed:
                dist.barrier()

                
            valT_top1, valT_top5,  valB_top1, valB_top5, val_losses, val_speed = validate1(val_loader, model_degrad, model_target1, model_budget1, val_criterion, step='target', gpu_id= args.gpu,  d=ctr)

            val_logger.log({'epoch': epoch,'prec1_T': valT_top1.item(), 'prec1_B': valB_top1.item()})

            scheduler.step()

            if args.distributed:
                dist.barrier()


            print('Train: [{:03d}/{:03d}]\tLoss: {:4.4f}\tTopT@1: {:.4f}\tTopB@1: {:.4f}\t'
                      'Speed: {:.2f} ms/batch\tData loading: {:.2f} ms/batch'.format(
                        epoch + 1, total_epochs, train_losses, trainT_top1, trainB_top1, train_speed * 1000.0,
                        speed_data_loader * 1000.0), flush=True)
               
            print('Val  : [{:03d}/{:03d}]\tLoss: {:4.4f}\tTopT@1: {:.4f}\tTopB@1: {:.4f}\t'
                      'Speed: {:.2f} ms/batch'.format(epoch + 1, total_epochs, val_losses, valT_top1,
                                                      valB_top1, val_speed * 1000.0),flush=True)




            is_best = valT_top1
            best_top1 = valT_top1
     
            
            save_dict = {'net': model_target1,
                         'epoch': epoch,
                         'state_dict': model_target1.state_dict(),
                         'acc': best_top1,
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict()
                        }

              
            torch.save(save_dict, save_dest+'/'+ 'model_target_'+str(ctr)+'.ckpt')        
        
        
        #----------------- END OF TARGET MODEL TRAINING------------------#
        
        

        #----------------- START OF BUDGET MODEL TRAINING------------------#
        
        train_logger = Logger(save_dest+'/'+'model_budget_train_'+str(ctr)+'.log',['epoch','prec1_T', 'prec1_B'])
        val_logger = Logger(save_dest+'/'+'model_budget_val_'+str(ctr)+'.log',['epoch','prec1_T', 'prec1_B'])

      
        sys.stdout.flush()
        checkpoint = torch.load(imagenet_path)
        model_budget1.load_state_dict(checkpoint['state_dict'], strict= False)        
        
        params = model_budget1.parameters()
        optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max= total_epochs, eta_min=1e-7, verbose=True)

        for epoch in range(args.start_epoch, total_epochs):
            
            print('\n Running Budget Model Now')
            
            trainT_top1, trainT_top5, trainB_top1, trainB_top5, train_losses, train_speed, speed_data_loader, train_steps = train1(train_loader, model_degrad, model_target1, model_budget1, optimizer,  
                                                                                                                  train_criterion, train_entropy_criterion,  epoch + 1, display =  
                                                                                                                 args.print_freq, clip_gradient=args.clip_gradient, gpu_id= args.gpu, rank=args.rank, step='budget',  d=ctr)

            train_logger.log({'epoch': epoch,'prec1_T': trainT_top1.item(), 'prec1_B': trainB_top1.item()})
            
            if args.distributed:
                dist.barrier()

                
            valT_top1, valT_top5,  valB_top1, valB_top5, val_losses, val_speed = validate1(val_loader, model_degrad, model_target1, model_budget1,val_criterion, step='budget', gpu_id= args.gpu,  d=ctr)


            val_logger.log({'epoch': epoch,'prec1_T': valT_top1.item(), 'prec1_B': valB_top1.item()})

            scheduler.step()

            if args.distributed:
                dist.barrier()

            
            print('Train: [{:03d}/{:03d}]\tLoss: {:4.4f}\tTopT@1: {:.4f}\tTopB@1: {:.4f}\t'
                      'Speed: {:.2f} ms/batch\tData loading: {:.2f} ms/batch'.format(
                        epoch + 1, total_epochs, train_losses, trainT_top1, trainB_top1, train_speed * 1000.0,
                        speed_data_loader * 1000.0), flush=True)
            
            print('Val  : [{:03d}/{:03d}]\tLoss: {:4.4f}\tTopT@1: {:.4f}\tTopB@1: {:.4f}\t'
                      'Speed: {:.2f} ms/batch'.format(epoch + 1, total_epochs, val_losses, valT_top1,
                                                      valB_top1, val_speed * 1000.0),flush=True)




            is_best = valT_top1
            best_top1 = valT_top1
            
             
            save_dict = {'net': model_budget1,
                         'epoch': epoch,
                         'state_dict': model_budget1.state_dict(),
                         'acc': best_top1,
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict()
                        }

            torch.save(save_dict, save_dest+'/'+ 'model_budget_'+str(ctr)+'.ckpt')
         
        #----------------- END OF BUDGET MODEL TRAINING------------------#
        





if __name__ == '__main__':
    main()
