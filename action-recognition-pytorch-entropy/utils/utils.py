import shutil
import os
import time
import multiprocessing
import numpy as np
import cv2
import torch.nn.functional as F
import torch
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tqdm import tqdm
from .video_transforms import (GroupRandomHorizontalFlip,
                               GroupMultiScaleCrop, GroupScale, GroupCenterCrop, GroupRandomCrop,
                               GroupNormalize, Stack, ToTorchFormatTensor, GroupRandomScale)
import random
from  torch.nn.modules.loss import _Loss
from PIL import Image, ImageOps
import imageio

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, 2)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



def save_checkpoint(state, is_best, filepath=''):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'),
                        os.path.join(filepath, 'model_best.pth.tar'))


def get_augmentor(is_train, image_size, mean=None,
                  std=None, disable_scaleup=False, is_flow=False,
                  threed_data=False, version='v1', scale_range=None):

    mean = [0.485, 0.456, 0.406] if mean is None else mean
    std = [0.229, 0.224, 0.225] if std is None else std
    scale_range = [256, 320] if scale_range is None else scale_range
    augments = []

    if is_train:
        if version == 'v1':
            augments += [
                GroupMultiScaleCrop(image_size, [1, .875, .75, .66])
            ]
        elif version == 'v2':
            augments += [
                GroupRandomScale(scale_range),
                GroupRandomCrop(image_size),
            ]
        augments += [GroupRandomHorizontalFlip(is_flow=is_flow)]
    else:
        scaled_size = image_size if disable_scaleup else int(image_size / 0.875 + 0.5)
        augments += [
            GroupScale(scaled_size),
            GroupCenterCrop(image_size)
        ]
    augments += [
        Stack(threed_data=threed_data),
        ToTorchFormatTensor(),
        GroupNormalize(mean=mean, std=std, threed_data=threed_data)
    ]

    augmentor = transforms.Compose(augments)
    return augmentor


def build_dataflow(dataset, is_train, batch_size, workers=36, is_distributed=False):
    workers = min(workers, multiprocessing.cpu_count())
    shuffle = False

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    if is_train:
        shuffle = sampler is None

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=workers, pin_memory=True, sampler=sampler)

    return data_loader






def train(data_loader, model_degrad, model_target, model_budget, optimizer_t, optimizer_b, train_criterion, train_entropy_criterion, epoch, step, display=100, steps_per_epoch=99999999999,  
    clip_gradient=None,  gpu_id=None, rank=0):


    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses_degrad= AverageMeter()
    losses_target= AverageMeter()
    losses_budget= AverageMeter()
    losses= AverageMeter()

    top1_target= AverageMeter()
    top5_target= AverageMeter()

    top1_budget= AverageMeter()
    top5_budget= AverageMeter()


    # set different random see every epoch
    if dist.is_initialized():
        data_loader.sampler.set_epoch(epoch)

    # fix encoder, and train target and privacy net

    model_degrad.train()
    model_target.train()
    model_budget.train()

    end = time.time()
    num_batch = 0


    with tqdm(total=len(data_loader)) as t_bar:
        for i, (images, target_actor, target_action) in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            # compute output
        
            images = images.cuda(gpu_id, non_blocking=True)
            target_action = target_action.cuda(gpu_id, non_blocking=True)
            target_actor = target_actor.cuda(gpu_id, non_blocking=True)

            loss_target = torch.tensor(0)
            loss_budget = torch.tensor(0)

            optimizer_t.zero_grad()


            output_degrad, bias = model_degrad(images)
            output_target = model_target(output_degrad)
            r = random.randint(0,output_degrad.size(2)-1)
            output_budget = model_budget(output_degrad[:,:,r,:,:]) 

            loss_target = train_criterion(output_target, target_action)
            entropy_budget = train_entropy_criterion(F.softmax(output_budget, dim=1))

            loss_t = loss_target-2*entropy_budget+10.

            loss_t.backward()
            optimizer_t.step()


            optimizer_b.zero_grad()


            output_degrad, bias = model_degrad(images)
            output_target = model_target(output_degrad)
            output_budget = model_budget(output_degrad[:,:,r,:,:]) 


            loss_budget = train_criterion(output_budget, target_actor) 

            loss_b = loss_budget

            loss_b.backward()
            optimizer_b.step()


            # measure accuracy and record loss
            prec1_target, prec5_target = accuracy(output_target, target_action)
            prec1_budget, prec5_budget = accuracy(output_budget, target_actor)

            losses_target.update(loss_target.item(), images.size(0))
            losses_budget.update(loss_budget.item(), images.size(0))
            losses_degrad.update(entropy_budget.item(), images.size(0))
            losses.update(entropy_budget.item(), images.size(0))

            top1_target.update(prec1_target[0], images.size(0))
            top5_target.update(prec5_target[0], images.size(0))
            top1_budget.update(prec1_budget[0], images.size(0))
            top5_budget.update(prec5_budget[0], images.size(0))



            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            num_batch += 1
            t_bar.update(1)
            if i > steps_per_epoch:
                break


    return top1_target.avg, top5_target.avg, top1_budget.avg, top5_budget.avg, losses.avg, batch_time.avg, data_time.avg, num_batch


def validate(data_loader, model_degrad, model_target, model_budget, train_criterion, gpu_id=None):


    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses_degrad= AverageMeter()
    losses_target= AverageMeter()
    losses_budget= AverageMeter()

    losses= AverageMeter()

    top1_target= AverageMeter()
    top5_target= AverageMeter()

    top1_budget= AverageMeter()
    top5_budget= AverageMeter()

    # switch to evaluate mode
    model_degrad.eval()
    model_target.eval()
    model_budget.eval()

    with torch.no_grad(), tqdm(total=len(data_loader)) as t_bar:
        end = time.time()

        for i, (images, target_actor, target_action) in enumerate(data_loader):

            images = images.cuda(gpu_id, non_blocking=True)

            target_action = target_action.cuda(gpu_id, non_blocking=True)
            target_actor = target_actor.cuda(gpu_id, non_blocking=True)

            loss_target = torch.tensor(0)
            loss_budget = torch.tensor(0)

            with torch.no_grad():
                output_degrad, bias = model_degrad(images)
                output_target = model_target(output_degrad)
                output_budget = []
                for r in range(output_degrad.size(2)):
                    output_budget.append(model_budget(output_degrad[:,:,r,:,:])) 

            loss_target = train_criterion(output_target, target_action)
            loss_budget = 0
            for r in range(output_degrad.size(2)):            
                loss_budget += train_criterion(output_budget[r], target_actor) 

            loss_budget /= output_degrad.size(2) 
            #loss_budget = e_criterion(output_budget) 

            loss = loss_target + loss_budget
            #loss = loss_budget

            # measure accuracy and record loss
            prec1_target, prec5_target = accuracy(output_target, target_action)
            prec1_budget= 0
            prec5_budget= 0
            for r in range(output_degrad.size(2)):            
                prec1_budget_, prec5_budget_ = accuracy(output_budget[r], target_actor)
                prec1_budget += prec1_budget_
                prec5_budget += prec5_budget_

            prec1_budget /= output_degrad.size(2) 
            prec5_budget /= output_degrad.size(2) 

            losses_target.update(loss_target.item(), images.size(0))
            losses_budget.update(loss_budget.item(), images.size(0))
            losses_degrad.update(loss.item(), images.size(0))
            losses.update(loss.item(), images.size(0))

            top1_target.update(prec1_target[0], images.size(0))
            top5_target.update(prec5_target[0], images.size(0))

            top1_budget.update(prec1_budget[0], images.size(0))
            top5_budget.update(prec5_budget[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            t_bar.update(1)
    print(bias)
    return top1_target.avg, top5_target.avg, top1_budget.avg, top5_budget.avg, losses.avg, batch_time.avg





def train1(data_loader, model_degrad, model_target1, model_budget1, optimizer, train_criterion, train_entropy_criterion, epoch, step, d, display=100, steps_per_epoch=99999999999,  
    clip_gradient=None,  gpu_id=None, rank=0):


    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses_degrad= AverageMeter()
    losses_target= AverageMeter()
    losses_budget= AverageMeter()
    losses= AverageMeter()

    top1_target= AverageMeter()
    top5_target= AverageMeter()

    top1_budget= AverageMeter()
    top5_budget= AverageMeter()


    # set different random see every epoch
    if dist.is_initialized():
        data_loader.sampler.set_epoch(epoch)

    # fix encoder, and train target and privacy net

    model_degrad.train()
    model_target1.train()
    model_budget1.train()
    #model_reconst.train()

    end = time.time()
    num_batch = 0


    with tqdm(total=len(data_loader)) as t_bar:
        for i, (images, target_actor, target_action) in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            # compute output
        
            images = images.cuda(gpu_id, non_blocking=True)
            target_action = target_action.cuda(gpu_id, non_blocking=True)
            target_actor = target_actor.cuda(gpu_id, non_blocking=True)

            loss_target = torch.tensor(0)
            loss_budget = torch.tensor(0)

            optimizer.zero_grad()

            with torch.no_grad():
                output_degrad, bias = model_degrad(images)

            output_target = model_target1(output_degrad)
            r = random.randint(0,output_degrad.size(2)-1)
            output_budget = model_budget1(output_degrad[:,:,r,:,:]) 

            loss_target = train_criterion(output_target, target_action)
            loss_budget = train_criterion(output_budget, target_actor) 

            if step == 'target':
                loss = loss_target
            elif step == 'budget':
                loss = loss_budget
            else:
                print("Invalid mode detected")
                exit()

            loss.backward()
            optimizer.step()


            # measure accuracy and record loss
            prec1_target, prec5_target = accuracy(output_target, target_action)
            prec1_budget, prec5_budget = accuracy(output_budget, target_actor)

            losses_target.update(loss_target.item(), images.size(0))
            losses_budget.update(loss_budget.item(), images.size(0))
            losses_degrad.update(loss.item(), images.size(0))
            losses.update(loss.item(), images.size(0))

            top1_target.update(prec1_target[0], images.size(0))
            top5_target.update(prec5_target[0], images.size(0))
            top1_budget.update(prec1_budget[0], images.size(0))
            top5_budget.update(prec5_budget[0], images.size(0))



            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            num_batch += 1
            t_bar.update(1)
            if i > steps_per_epoch:
                break


    return top1_target.avg, top5_target.avg, top1_budget.avg, top5_budget.avg, losses.avg, batch_time.avg, data_time.avg, num_batch


def validate1(data_loader, model_degrad, model_target1, model_budget1, train_criterion, step,d, gpu_id=None):


    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses_degrad= AverageMeter()
    losses_target= AverageMeter()
    losses_budget= AverageMeter()

    losses= AverageMeter()

    top1_target= AverageMeter()
    top5_target= AverageMeter()

    top1_budget= AverageMeter()
    top5_budget= AverageMeter()

    # switch to evaluate mode
    model_degrad.eval()
    model_target1.eval()
    model_budget1.eval()


    with torch.no_grad(), tqdm(total=len(data_loader)) as t_bar:
        end = time.time()

        for i, (images, target_actor, target_action) in enumerate(data_loader):

            images = images.cuda(gpu_id, non_blocking=True)

            target_action = target_action.cuda(gpu_id, non_blocking=True)
            target_actor = target_actor.cuda(gpu_id, non_blocking=True)

            loss_target = torch.tensor(0)
            loss_budget = torch.tensor(0)

            with torch.no_grad():
                output_degrad, bias = model_degrad(images)
                output_target = model_target1(output_degrad)

                
                output_budget = []
                for r in range(output_degrad.size(2)):
                    output_budget.append(model_budget1(output_degrad[:,:,r,:,:])) 
               

            loss_target = train_criterion(output_target, target_action)
            
            loss_budget = 0
            for r in range(output_degrad.size(2)):            
                loss_budget += train_criterion(output_budget[r], target_actor) 
            

            loss_budget /= output_degrad.size(2) 
            #loss_budget = e_criterion(output_budget) 

            if step == 'target':
                loss = loss_target
            elif step == 'budget':
                loss = loss_budget
            else:
                print("Invalid mode detected")
                exit()

            # measure accuracy and record loss
            prec1_target, prec5_target = accuracy(output_target, target_action)
            prec1_budget, prec5_budget = accuracy(output_budget, target_actor)

            
            prec1_budget= 0
            prec5_budget= 0

            for r in range(output_degrad.size(2)):
                #print(r)            
                prec1_budget_, prec5_budget_ = accuracy(output_budget[r], target_actor)
                prec1_budget += prec1_budget_
                prec5_budget += prec5_budget_

            prec1_budget /= output_degrad.size(2) 
            prec5_budget /= output_degrad.size(2) 
            


            losses_target.update(loss_target.item(), images.size(0))
            losses_budget.update(loss_budget.item(), images.size(0))
            losses_degrad.update(loss.item(), images.size(0))
            losses.update(loss.item(), images.size(0))

            top1_target.update(prec1_target[0], images.size(0))
            top5_target.update(prec5_target[0], images.size(0))

            top1_budget.update(prec1_budget[0], images.size(0))
            top5_budget.update(prec5_budget[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            t_bar.update(1)
    print(bias)
    return top1_target.avg, top5_target.avg, top1_budget.avg, top5_budget.avg, losses.avg, batch_time.avg






























'''
def train_target(data_loader, model_degrad, model_target, model_budget, optimizer_degrad, optimizer_target, optimizer_budget, criterion, epoch, display=100, steps_per_epoch=99999999999,  
    clip_gradient=None,  gpu_id=None, rank=0, alpha=0.9):

    #model_degrad= model[0]
    #model_target= model[1]

    #optimizer_target= optimizer[1]

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses_target= AverageMeter()
    top1_target= AverageMeter()
    top5_target= AverageMeter()
    s_losses_target= AverageMeter()


    # set different random see every epoch
    if dist.is_initialized():
        data_loader.sampler.set_epoch(epoch)

    # switch to train model

    model_degrad.train()
    model_target.train()
    
    iteration = 0

    end = time.time()
    num_batch = 0


    with tqdm(total=len(data_loader)) as t_bar:
        for i, (images, target_actor, target_action) in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            # compute output
            #if gpu_id is not None:
            images = images.cuda(gpu_id, non_blocking=True)

            target_action = target_action.cuda(gpu_id, non_blocking=True)
            target_actor = target_actor.cuda(gpu_id, non_blocking=True)


            #with torch.no_grad():           
            output_degrad = model_degrad(images)

            #print(sud)
            output_target, output_target_soft = model_target(output_degrad)

            loss_target = criterion(output_target, target_action)

            optimizer_degrad.zero_grad()
            optimizer_target.zero_grad()
            loss_target.backward()
            optimizer_degrad.step()
            optimizer_target.step()


            # measure accuracy and record loss
            prec1_target, prec5_target = accuracy(output_target, target_action)
            losses_target.update(loss_target.item(), images.size(0))
            top1_target.update(prec1_target[0], images.size(0))
            top5_target.update(prec5_target[0], images.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % display == 0 and rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(data_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses_target, top1=top1_target, top5=top5_target), flush=True)
            num_batch += 1
            t_bar.update(1)
            if i > steps_per_epoch:
                break

    return top1_target.avg, top5_target.avg, losses_target.avg, batch_time.avg, data_time.avg, num_batch



def val_target(data_loader, model_degrad, model_target, model_budget, criterion, gpu_id=None, alpha=0.9):

    #model_degrad= model[0]
    #model_target= model[1]

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses_target= AverageMeter()
    top1_target= AverageMeter()
    top5_target= AverageMeter()


    # set different random see every epoch
    if dist.is_initialized():
        data_loader.sampler.set_epoch(epoch)

    # switch to train model

    model_degrad.eval()
    model_target.eval()
    
    iteration = 0

    end = time.time()
    num_batch = 0


    with tqdm(total=len(data_loader)) as t_bar:
        for i, (images, target_actor, target_action) in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            # compute output
            #if gpu_id is not None:
            images = images.cuda(gpu_id, non_blocking=True)

            target_action = target_action.cuda(gpu_id, non_blocking=True)
            target_actor = target_actor.cuda(gpu_id, non_blocking=True)


            #with torch.no_grad():
            output_degrad = model_degrad(images)

            output_target, output_target_soft = model_target(output_degrad)
            loss_target = criterion(output_target, target_action)


            # measure accuracy and record loss
            prec1_target, prec5_target = accuracy(output_target, target_action)
            losses_target.update(loss_target.item(), images.size(0))
            top1_target.update(prec1_target[0], images.size(0))
            top5_target.update(prec5_target[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            t_bar.update(1)

    return top1_target.avg, top5_target.avg, losses_target.avg, batch_time.avg





def train_budget(data_loader, model_degrad, model_target, model_budget, optimizer_degrad, optimizer_target, optimizer_budget, criterion, epoch, display=100, steps_per_epoch=99999999999,  
    clip_gradient=None,  gpu_id=None, rank=0, alpha=0.9):

    #model_degrad= model[0]
    #model_budget= model[2]

    #optimizer_budget= optimizer[2]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_budget= AverageMeter()
    top1_budget= AverageMeter()
    top5_budget= AverageMeter()


    # set different random see every epoch
    if dist.is_initialized():
        data_loader.sampler.set_epoch(epoch)

    # switch to train model

    model_degrad.eval()
    model_budget.train()
    
    iteration = 0

    end = time.time()
    num_batch = 0


    with tqdm(total=len(data_loader)) as t_bar:
        for i, (images, target_actor, target_action) in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            # compute output
            
            #if gpu_id is not None:
            images = images.cuda(gpu_id, non_blocking=True)

            target_action = target_action.cuda(gpu_id, non_blocking=True)
            target_actor = target_actor.cuda(gpu_id, non_blocking=True)


            with torch.no_grad():
                x = images
                y=x[0,:,:,:,:]
                for i in range(16):
                    y= x[0,:,i,:,:]
                    y=np.transpose((y.cpu().data.numpy()), (1,2,0))
                    print(y.shape)
                    img= y[:,:,:]
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    cv2.imshow("image", gray)
                    cv2.waitKey()
                                   
                output_degrad = model_degrad(images)

                x = output_degrad
                y=x[0,:,:,:,:]
                for i in range(16):
                    y= x[0,:,i,:,:]
                    y=np.transpose((y.cpu().data.numpy()), (1,2,0))
                    print(y.shape)
                    img= y[:,:,:]
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    #gray = cv2.threshold(gray*255, 127, 255, cv2.THRESH_BINARY)[1]
                    cv2.imshow("image", (gray))
                    #gray = cv2.convertScaleAbs(gray, alpha=(255.0))
                    #cv2.imwrite('sbuMEr-'+str(i)+'.png',gray*255.)
                    cv2.waitKey()

            print(sud)
            output_budget, output_budget_soft = model_budget(output_degrad.detach())
            loss_budget = criterion(output_budget, target_actor)

            optimizer_budget.zero_grad()
            loss_budget.backward()
            optimizer_budget.step()


            # measure accuracy and record loss
            prec1_budget, prec5_budget = accuracy(output_budget, target_actor)
            losses_budget.update(loss_budget.item(), images.size(0))
            top1_budget.update(prec1_budget[0], images.size(0))
            top5_budget.update(prec5_budget[0], images.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % display == 0 and rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(data_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses_budget, top1=top1_budget, top5=top5_budget), flush=True)
            num_batch += 1
            t_bar.update(1)
            if i > steps_per_epoch:
                break

    return top1_budget.avg, top5_budget.avg, losses_budget.avg, batch_time.avg, data_time.avg, num_batch



def val_budget(data_loader, model_degrad, model_target, model_budget, criterion, gpu_id=None, alpha=0.1):

    #model_degrad= model[0]
    #model_budget= model[2]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_budget= AverageMeter()
    top1_budget= AverageMeter()
    top5_budget= AverageMeter()


    # set different random see every epoch
    if dist.is_initialized():
        data_loader.sampler.set_epoch(epoch)

    # switch to train model

    model_degrad.eval()
    model_budget.eval()
    
    iteration = 0

    end = time.time()
    num_batch = 0


    with tqdm(total=len(data_loader)) as t_bar:
        for i, (images, target_actor, target_action) in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            # compute output
            #if gpu_id is not None:
            images = images.cuda(gpu_id, non_blocking=True)

            target_action = target_action.cuda(gpu_id, non_blocking=True)
            target_actor = target_actor.cuda(gpu_id, non_blocking=True)

            with torch.no_grad():
                output_degrad = model_degrad(images)

         
            output_budget, output_budget_soft = model_budget(output_degrad.detach())
            loss_budget = criterion(output_budget, target_actor)


            # measure accuracy and record loss
            prec1_budget, prec5_budget = accuracy(output_budget, target_actor)
            losses_budget.update(loss_budget.item(), images.size(0))
            top1_budget.update(prec1_budget[0], images.size(0))
            top5_budget.update(prec5_budget[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            t_bar.update(1)

    return top1_budget.avg, top5_budget.avg, losses_budget.avg, batch_time.avg
'''



