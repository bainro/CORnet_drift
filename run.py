import os, argparse, time, glob, pickle, subprocess, shlex, io, pprint

import numpy as np
import pandas
import tqdm
import fire

import torch
import torch.nn as nn
import torch.utils.model_zoo
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision

import cornet

from PIL import Image
Image.warnings.simplefilter('ignore')

np.random.seed(0)
torch.manual_seed(0)

torch.backends.cudnn.benchmark = True
normalize = torchvision.transforms.Normalize((0.5, 0.5, 0.5), 
                                             (0.5, 0.5, 0.5))

parser = argparse.ArgumentParser(description='CIFAR100 Training')
parser.add_argument('-o', '--output_path', default=None,
                    help='path for storing ')
parser.add_argument('--dropout', action='store_true',
                    help='whether to add dropout to all layers or not')
parser.add_argument('--model', choices=['Z', 'R', 'RT', 'S'], default='Z',
                    help='which model to train')
parser.add_argument('--times', default=5, type=int,
                    help='number of time steps to run the model (only R model)')
parser.add_argument('--ngpus', default=0, type=int,
                    help='number of GPUs to use; 0 if you want to run on CPU')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers')
parser.add_argument('--epochs', default=1, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=128, type=int,
                    help='mini-batch size')
parser.add_argument('--lr', '--learning_rate', default=1e-1, type=float,
                    help='initial learning rate')
parser.add_argument('--step_size', default=30, type=int,
                    help='after how many epochs learning rate should be decreased 10x')
parser.add_argument('--momentum', default=.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay ')


FLAGS, FIRE_FLAGS = parser.parse_known_args()
  

def get_model(pretrained=False):
    map_location = None if FLAGS.ngpus > 0 else 'cpu'
    model = getattr(cornet, f'cornet_{FLAGS.model.lower()}')
    if FLAGS.model.lower() == 'r':
        model = model(pretrained=pretrained, map_location=map_location, times=FLAGS.times)
    else:
        model = model(pretrained=pretrained, map_location=map_location, dropout=FLAGS.dropout)

    if FLAGS.ngpus == 0:
        model = model.module  # remove DataParallel
    if FLAGS.ngpus > 0:
        model = model.cuda()
    return model


def train(restore_path=None,  # useful when you want to restart training
          save_train_epochs=.1,  # how often save output during training
          save_val_epochs=1.,  # how often save output during validation
          save_model_epochs=5,  # how often save model weigths
          save_model_secs=60 * 10  # how often save model (in sec)
          ):

    model = get_model()
    trainer = CIFAR100Train(model)
    validator = CIFAR100Val(model)

    start_epoch = 0
    if restore_path is not None:
        ckpt_data = torch.load(restore_path)
        start_epoch = ckpt_data['epoch']
        model.load_state_dict(ckpt_data['state_dict'])
        trainer.optimizer.load_state_dict(ckpt_data['optimizer'])

    records = []
    recent_time = time.time()

    nsteps = len(trainer.data_loader)
    if save_train_epochs is not None:
        save_train_steps = (np.arange(0, FLAGS.epochs + 1,
                                      save_train_epochs) * nsteps).astype(int)
    if save_val_epochs is not None:
        save_val_steps = (np.arange(0, FLAGS.epochs + 1,
                                    save_val_epochs) * nsteps).astype(int)
    if save_model_epochs is not None:
        save_model_steps = (np.arange(0, FLAGS.epochs + 1,
                                      save_model_epochs) * nsteps).astype(int)

    results = {'meta': {'step_in_epoch': 0,
                        'epoch': start_epoch,
                        'wall_time': time.time()}
               }
    for epoch in tqdm.trange(0, FLAGS.epochs + 1, initial=start_epoch, desc='epoch'):
        data_load_start = np.nan
        for step, data in enumerate(tqdm.tqdm(trainer.data_loader, desc=trainer.name)):
            data_load_time = time.time() - data_load_start
            global_step = epoch * len(trainer.data_loader) + step

            if save_val_steps is not None:
                if global_step in save_val_steps:
                    results[validator.name] = validator()
                    trainer.model.train()

            if FLAGS.output_path is not None:
                records.append(results)
                if len(results) > 1:
                    pickle.dump(records, open(os.path.join(FLAGS.output_path, 'results.pkl'), 'wb'))

                ckpt_data = {}
                ckpt_data['flags'] = FLAGS.__dict__.copy()
                ckpt_data['epoch'] = epoch
                ckpt_data['state_dict'] = model.state_dict()
                ckpt_data['optimizer'] = trainer.optimizer.state_dict()

                if save_model_secs is not None:
                    if time.time() - recent_time > save_model_secs:
                        torch.save(ckpt_data, os.path.join(FLAGS.output_path,
                                                           'latest_checkpoint.pth.tar'))
                        recent_time = time.time()

                if save_model_steps is not None:
                    if global_step in save_model_steps:
                        torch.save(ckpt_data, os.path.join(FLAGS.output_path,
                                                           f'epoch_{epoch:02d}.pth.tar'))

            else:
                if len(results) > 1:
                    pprint.pprint(results)

            if epoch < FLAGS.epochs:
                frac_epoch = (global_step + 1) / len(trainer.data_loader)
                record = trainer(frac_epoch, *data)
                # record['data_load_dur'] = data_load_time
                results = {'meta': {'step_in_epoch': step + 1,
                                    'epoch': frac_epoch,
                                    'wall_time': time.time()}
                           }
                if save_train_steps is not None:
                    if step in save_train_steps:
                        results[trainer.name] = record

            data_load_start = time.time()
        

def train_movie_test(num_epochs=10, 
                     num_movies=1, # how many times to sample from the movie split
                     restore_path=None): # where to load the pretrained model from
    """
    Train, movie, test loop until num_epochs of train split has been trained on.
    Sample from each CORnet layer every 1/10th training epoch using movie split.
    Then evaluate the model on test split to see if behavior changes over time.
    """
    model = get_model()
    trainer = CIFAR100Train(model)
    validator = CIFAR100Val(model, movie=True)
    
    assert restore_path is not None, "set restore_path"    
    ckpt_data = torch.load(restore_path)
    model.load_state_dict(ckpt_data['state_dict'])
    # Goldilock's LR = 1e-3 (from results.pkl) not too small nor too large
    ckpt_data['optimizer']['param_groups'][0]['lr'] = 0.001
    trainer.optimizer.load_state_dict(ckpt_data['optimizer'])
    
    a_tenth = len(trainer.data_loader) // 10
    # holds all samples from the model's layers when running on movie
    # gets saved to a pandas dataframe
    model_feats = None

    """ learn on train set for 1/10th of an epoch. """
    for epoch in range(0, num_epochs):
        # only train on 1/10th & start again where left off
        for i, (x, targets) in enumerate(trainer.data_loader):
            model.train()
            if FLAGS.ngpus > 0:
                targets = targets.cuda(non_blocking=True)
            output = model(x)
            loss = trainer.loss(output, targets)
            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()
            
            if i % a_tenth == 0 and i != 0:
                """ train on movie for (10 repeats or just once) while sampling layers' neurons """
                
                # layers (choose from: V1, V2, V4, IT, decoder)
                # sublayer (e.g., output, conv1, avgpool)
                def _store_feats(sublayer, inp, output):
                    # way of accessing intermediate model features
                    output = output.detach().cpu().numpy()
                    _model_feats.append(output)
                  
                def pairwise(iterable):
                    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
                    a = iter(iterable)
                    return zip(a, a)
   
                for repeat in range(num_movies):
                    hook_handles = []
                    # grab the output of each layer's non-linearity (ie ReLU)
                    for layer, sublayer in zip(["V1", "V2", "V4", "IT"], ["nonlin"] * 4):
                        model_layer = getattr(getattr(model.module, layer), sublayer)
                        hook_handle = model_layer.register_forward_hook(_store_feats)
                        hook_handles.append(hook_handle)
                    for (x, targets) in validator.movie_loader:
                        _model_feats = []
                        bs_flats = None
                        if FLAGS.ngpus > 0:
                            targets = targets.cuda(non_blocking=True)
                        output = model(x)     
                        # THIS IS NOT GENERAL! Specific to training on 2 GPUs
                        sorted_model_feats = []
                        # hooks are async & can return in mixed order so must sort
                        for tensor in _model_feats:
                            # find idx into sorted_model_feats that tensor belongs
                            if len(sorted_model_feats) == 0:
                                sorted_model_feats.append(tensor)
                            else:
                                broke_out = False
                                for j in range(len(sorted_model_feats)):
                                    num_conv_kernels = tensor.shape[1]
                                    j_num_conv_kernels = sorted_model_feats[j].shape[1]
                                    if num_conv_kernels <= j_num_conv_kernels:
                                        sorted_model_feats.insert(j, tensor)
                                        broke_out = True
                                        break
                                if not broke_out:
                                    sorted_model_feats.append(tensor)
                            
                        for tensor_gpu1, tensor_gpu2 in pairwise(sorted_model_feats):
                            # (batchsize, C * W * H)
                            bs_flat_1 = np.reshape(tensor_gpu1, (tensor_gpu1.shape[0], -1))
                            bs_flat_2 = np.reshape(tensor_gpu2, (tensor_gpu2.shape[0], -1))
                            # (2 * batchsize, C * W * H)
                            _tmp = np.array([])
                            bs_flat = np.append(_tmp, np.vstack((bs_flat_1, bs_flat_2)))
                            if type(bs_flats) == type(None):
                                bs_flats = bs_flat
                            else:
                                bs_flats = np.hstack((bs_flats, bs_flat))
                        
                        if type(model_feats) == type(None):
                            model_feats = bs_flats
                        else:
                            model_feats = np.vstack((model_feats, bs_flats))
                        loss = trainer.loss(output, targets)
                        trainer.optimizer.zero_grad()
                        loss.backward()
                        trainer.optimizer.step()
                    """ save output file for each movie repeat """
                    num_tenths_this_epoch = i // a_tenth
                    mov_r = (num_movies * epoch * 10) + repeat + num_tenths_this_epoch
                    # to avoid OOM issues!
                    for handle in hook_handles:
                        handle.remove()
                    """ evaluate test set accuracy without learning """
                    test_acc = validator()["top1"]
                    print(f"test accuracy: {test_acc * 100:.1f}%")
                    np.save(os.path.join(FLAGS.output_path, f"movie_{mov_r}_e_{epoch+1}_test_{test_acc * 100:.1f}"), model_feats)
                    print(f"model_feats.shape: {model_feats.shape}")
                    # reset since just saved
                    model_feats = None

    print("\ntrain_movie_test() done!!!\n")
        
        
class CIFAR100Train(object):

    def __init__(self, model):
        self.name = 'train'
        self.model = model
        self.data_loader = self.data()
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         FLAGS.lr,
                                         momentum=FLAGS.momentum,
                                         weight_decay=FLAGS.weight_decay)
        self.lr = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=FLAGS.step_size)
        self.loss = nn.CrossEntropyLoss()
        if FLAGS.ngpus > 0:
            self.loss = self.loss.cuda()

    def data(self):
        transform = torchvision.transforms.Compose([
                        torchvision.transforms.RandomResizedCrop(32),
                        torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.ToTensor(),
                        normalize,
                    ])
        dataset = torchvision.datasets.CIFAR100(root='./cifar100', train=True,
                                        download=True, transform=transform)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=FLAGS.batch_size,
                                                  shuffle=True,
                                                  num_workers=FLAGS.workers,
                                                  pin_memory=True)
        return data_loader

    def __call__(self, frac_epoch, inp, target):
        start = time.time()

        self.lr.step(epoch=frac_epoch)
        if FLAGS.ngpus > 0:
            target = target.cuda(non_blocking=True)
        output = self.model(inp)

        record = {}
        loss = self.loss(output, target)
        record['loss'] = loss.item()
        record['top1'] = accuracy(output, target)[0]
        record['top1'] /= len(output)
        record['learning_rate'] = self.lr.get_lr()[0]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        record['dur'] = time.time() - start
        return record


class CIFAR100Val(object):

    def __init__(self, model, movie=False):
        self.name = 'val'
        self.model = model
        self.test_loader, self.movie_loader = self.data(movie)
        self.loss = nn.CrossEntropyLoss(size_average=False)
        if FLAGS.ngpus > 0:
            self.loss = self.loss.cuda()

    def data(self, movie):
        # split test (10k) into test (9.8k) & movie (0.2k)
        shuffle = True
        movie_size = 0.02 if movie else 0.
        random_seed = 42
        data_dir = "./cifar100"

        transform = torchvision.transforms.Compose([
            #torchvision.transforms.Resize(36),
            #torchvision.transforms.CenterCrop(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        test_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False,
            download=True, transform=transform,
        )

        movie_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False,
            download=True, transform=transform,
        )

        num_test = len(test_dataset)
        indices = list(range(num_test))
        split = int(np.floor(movie_size * num_test))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        test_idx, movie_idx = indices[split:], indices[:split]
        test_sampler = SubsetRandomSampler(test_idx)
        movie_sampler = SubsetRandomSampler(movie_idx)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=FLAGS.batch_size, sampler=test_sampler,
            shuffle=False, num_workers=FLAGS.workers, pin_memory=True,
        )
        test_loader.num_images = len(test_idx)
        
        # Each forward pass recalc's dropout. Geometry did not use const mask across "movie" repeat
        movie_loader = torch.utils.data.DataLoader(
            movie_dataset, batch_size=FLAGS.batch_size, sampler=movie_sampler,
            shuffle=False, num_workers=FLAGS.workers, pin_memory=True,
        )
        movie_loader.num_images = len(movie_idx)

        return test_loader, movie_loader

    def __call__(self):
        self.model.eval()
        start = time.time()
        record = {'loss': 0, 'top1': 0}
        with torch.no_grad():
            for (inp, target) in tqdm.tqdm(self.test_loader, desc=self.name):
                if FLAGS.ngpus > 0:
                    target = target.cuda(non_blocking=True)
                output = self.model(inp)

                record['loss'] += self.loss(output, target).item()
                p1 = accuracy(output, target)
                record['top1'] += p1[0]
                
        # assert num_test_imgs == 10000, f'CIFAR100 should have 10,000 test images, not {num_test_imgs}'
        for key in record:
            record[key] /= self.test_loader.num_images
        record['dur'] = (time.time() - start) / len(self.test_loader)

        return record


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = [correct[:k].sum().item() for k in topk]
        return res


if __name__ == '__main__':
    fire.Fire(command=FIRE_FLAGS)
