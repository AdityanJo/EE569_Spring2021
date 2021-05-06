# Adityan Jothi
# USC ID 8162222801
# jothi@usc.edu

from model import ClassifierLN5, CustomClassifier

import torch 
from torchvision.datasets import CIFAR10 as CIFARDataset
from torchvision.datasets import MNIST as MNISTDataset
from torchvision.datasets import FashionMNIST as FashMNISTDataset
from torch.cuda import is_available as check_device_availability
from torch import device as Arch
from torch.utils.tensorboard import SummaryWriter as Logger
from torch import set_grad_enabled as disable_backprop
from torch import tensor as Tensor
from torch import norm as Norm
from torch import max as Max
from torch import save as Saver
from torchvision.transforms import Compose as BuildTransform
from torchvision.transforms import Resize as Rescale
from torchvision.transforms import ToTensor as ConvertToTensor
from torchvision.transforms import Normalize as AdjustNormal
from torch.utils.data import random_split as SplitRandomly
from torch.utils.data import DataLoader as DataGenerator
from torch.nn import CrossEntropyLoss as CELoss
from torch.optim import Adam as AdamOptimizer
from torch.optim import RMSprop as RMSOptimizer
from torch.optim import SGD as SGDOptimizer
from torch.utils.data import ConcatDataset as MergeDatasets

from os.path import join as pjoin
from os import makedirs as createdirectories
import numpy as np
import json

from copy import deepcopy as Copier
from tqdm import tqdm
from PIL.ImageOps import invert as ReversePixels
import matplotlib.pyplot as plt
import random
from utils import configuration

#Custom transform for inverting image for negatives dataset
class Reverse(object):
    def __init__(self):
        pass
    def __call__(self, x):
        H,W = x.size[:2]
        x = ReversePixels(x)
        return x

#Core training routine
def train(configuration):
    #Setup directories for checkpoints and tensorboard logs
    createdirectories(configuration.get('model_path'), exist_ok=True)
    createdirectories(pjoin(configuration.get('model_path'), 'logs'), exist_ok=True)
    createdirectories(pjoin(configuration.get('model_path'), 'checkpoints'), exist_ok=True)
    
    #Initialize tensorboard logger
    if configuration.get('enable_log'):
        logger = Logger(pjoin(configuration.get('model_path'),'logs'))
    
    #Choose target architecture
    if check_device_availability() and configuration.get('use_gpu'):
        target_arch = Arch('cuda:0')
    else:
        target_arch = Arch('cpu')

    network = configuration.get('network').to(target_arch)

    phases = configuration.get('phases')
    loaders = configuration.get('loaders')
    func_loss = configuration.get('loss')
    learner = configuration.get('learner')
    best_model_wts = Copier(network.state_dict())

    best_loss_val = 1e5
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    #Main Epoch Loop
    for epoch_num in range(configuration.get('max_epoch_val')):
        for phase in phases:
            if phase=='train':
                network.train()
            else:
                network.eval()
            
            if not loaders.get(phase):
                print(f'Not able to find loaders for phase: {phase}')
                continue
            
            loss_val = 0.0
            corrects_val = 0
            items_val = 0

            for idx, (images, targets) in tqdm(enumerate(loaders[phase])):
                images = images.to(target_arch)
                targets = targets.to(target_arch)

                learner.zero_grad()

                with disable_backprop(phase=='train'):
                    predictions = network(images)

                    loss_ = func_loss(predictions, targets)

                    if configuration.get('use_gpu'):
                        l2_regularization = Tensor(0.).cuda()
                    else:
                        l2_regularization = Tensor(0.)

                    
                    for param in network.parameters():
                        l2_regularization += Norm(param, 2)**2
                    
                    loss_ += 1e-3 * l2_regularization

                    if phase=='train':
                        loss_.backward()
                        learner.step()
                loss_val += loss_.item()
                items_val += images.size(0)
                _, class_preds = Max(predictions, 1)
                corrects_val += (targets==class_preds).sum().item()

            epoch_loss = loss_val/items_val
            epoch_acc = corrects_val/items_val
            
            if phase=='train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                test_loss.append(epoch_loss)
                test_acc.append(epoch_acc)
            
            #Tensorboard logging
            logger.add_scalar(phase+'_loss', epoch_loss, global_step=epoch_num)
            logger.add_scalar(phase+'_acc', epoch_acc, global_step=epoch_num)
            
            
            print(f'Epoch {epoch_num} [{phase}]: Loss = {epoch_loss}, Accuracy = {epoch_acc}')

            #Retain best weights copy for final save based on test set performance
            if phase=='test':
                if epoch_loss<best_loss_val:
                    best_loss_val = epoch_loss
                    best_model_wts = Copier(network.state_dict())
                if epoch_acc>configuration.get('acc_threshold'):
                    epoch_num = configuration.get('max_epoch_val')+1
                    break
        
        # Save checkpoint every 10 epochs
        if(epoch_num+1)%10==0:
            Saver(
                network.state_dict(),
                pjoin(
                    configuration.get('model_path'),
                    'checkpoints/model_'+str(epoch_num)+'.pth'
                )
            )
        #Stop training if reached limit
        if epoch_num>configuration.get('max_epoch_val'):
            break

    #Save final model
    network.load_state_dict(best_model_wts)
    Saver(
        network.state_dict(),
        pjoin(
            configuration.get('model_path'),
            'checkpoints/model_final.pth'
        )
    )
    return train_loss, train_acc, test_loss, test_acc

#Helper for drawing and storing results for problem1b
def drawGraphs1b(results, dataset_name):
    for idx in range(5):
        running_train_max_acc = []
        running_train_min_loss = []
        running_test_max_acc = []
        running_test_min_loss = []
        for jdx, run in results[f'config{idx+1}']['graphs']:
            train_loss, train_acc, test_loss, test_acc = run
            running_train_max_acc += [train_acc[np.argmax(test_acc)]]
            running_train_min_loss += [train_loss[np.argmax(test_acc)]]
            running_test_max_acc += [test_acc[np.argmax(test_acc)]]
            running_test_min_loss += [test_loss[np.argmax(test_acc)]]
            plt.figure()
            plt.plot(train_loss, label='train')
            plt.plot(test_loss, label='test')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Loss (Config{idx+1}-Run{jdx+1})')
            plt.legend()
            plt.savefig(f'{dataset_name}_loss_config{idx+1}_run{jdx+1}.png')
            plt.figure()
            plt.plot(train_acc, label='train')
            plt.plot(test_acc, label='test')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title(f'Accuracy (Config{idx+1}-Run{jdx+1})')
            plt.legend()
            plt.savefig(f'{dataset_name}_train_acc_config{idx+1}_run{jdx+1}.png')
        print(f'Config {idx+1}: {dataset_name}')
        print({
            'mean_train_loss': np.mean(running_train_min_loss),
            'mean_train_acc': np.mean(running_train_max_acc),
            'mean_test_loss': np.mean(running_test_min_loss),
            'mean_test_acc': np.mean(running_test_max_acc),
            'std_train_loss': np.std(running_train_min_loss),
            'std_train_acc': np.std(running_train_max_acc),
            'std_test_loss': np.std(running_test_min_loss),
            'std_test_acc': np.std(running_test_max_acc),
            
        })
        print('========')

#Driver function for running all configurations and runs for problem1b for a given dataset
def problem1b(train_dataset, test_dataset, dataset_name, acc_threshold):
    #Final results structure
    results = {
        'config1':{
            'graphs':None,
            'test_acc_mean':None,
            'test_acc_std':None,
        },
        'config2':{
            'graphs':None,
            'test_acc_mean':None,
            'test_acc_std':None,
        },
        'config3':{
            'graphs':None,
            'test_acc_mean':None,
            'test_acc_std':None,
        },
        'config4':{
            'graphs':None,
            'test_acc_mean':None,
            'test_acc_std':None,
        },
        'config5':{
            'graphs':None,
            'test_acc_mean':None,
            'test_acc_std':None,
        },

    }
    
    #Setting 1
    configuration['batch_size']=64
    train_iterator = DataGenerator(
        train_dataset,
        shuffle=True,
        batch_size=configuration.get('batch_size'),
        num_workers=configuration.get('num_workers')
    )
    test_iterator = DataGenerator(
        test_dataset,
        shuffle=False,
        batch_size=configuration.get('batch_size'),
        num_workers=configuration.get('num_workers')
    )
    
    if dataset_name=='cifar':
        #Different setting 1 for CIFAR
        test_acc_config1=[]
        graphs = []
        for i in range(5):
            network = ClassifierLN5(in_channels=1, number_of_classes=configuration.get('number_of_classes'), init_method='uniform')
            model_configuration={
                'enable_log' : configuration.get('enable_log'),
                'model_path': pjoin(configuration.get('model_path'), dataset_name, 'config1'),
                'network' : network,
                'use_gpu': configuration.get('use_gpu'),
                'phases': ['train','test'],
                'loaders':{
                    'train': train_iterator,
                    'test':test_iterator
                },
                'loss': CELoss(),
                'learner': RMSOptimizer(network.parameters(), lr=1e-3, weight_decay=0),
                'scheduler': None,
                'max_epoch_val': 50,
                'initialization_method': configuration.get('initialization_method'),
                'acc_threshold': acc_threshold,
            }

            train_loss, train_acc, test_loss, test_acc = train(model_configuration)
            graphs.append({
                'train_loss':train_loss,
                'train_acc':train_acc,
                'test_loss':test_loss,
                'test_acc': test_acc
            })
            max_test_acc = max(test_acc)
            print('Max test accuracy [Config 1]:', max_test_acc)
            test_acc_config1.append(max_test_acc)
        test_acc_config1 = np.array(test_acc_config1)
        print('Mean test accuracy [Config 1]:', np.mean(test_acc_config1))
        print('Std dev test accuracy [Config 1]:', np.std(test_acc_config1))
        results['config1']['graphs']=graphs
        results['config1']['test_acc_mean']=np.mean(test_acc_config1)
        results['config1']['test_acc_std']=np.std(test_acc_config1)
    else:
        #Different setting 1 for MNIST/Fashion MNIST
        test_acc_config1=[]
        graphs = []
        for i in range(5):
            network = ClassifierLN5(in_channels=1, number_of_classes=configuration.get('number_of_classes'), init_method='uniform')
            model_configuration={
                'enable_log' : configuration.get('enable_log'),
                'model_path': pjoin(configuration.get('model_path'), dataset_name, 'config1'),
                'network' : network,
                'use_gpu': configuration.get('use_gpu'),
                'phases': ['train','test'],
                'loaders':{
                    'train': train_iterator,
                    'test':test_iterator
                },
                'loss': CELoss(),
                'learner': AdamOptimizer(network.parameters(), lr=1e-3, weight_decay=0),
                'scheduler': None,
                'max_epoch_val': 50,
                'initialization_method': configuration.get('initialization_method'),
                'acc_threshold': acc_threshold,
            }

            train_loss, train_acc, test_loss, test_acc = train(model_configuration)
            graphs.append({
                'train_loss':train_loss,
                'train_acc':train_acc,
                'test_loss':test_loss,
                'test_acc': test_acc
            })
            max_test_acc = max(test_acc)
            print('Max test accuracy [Config 1]:', max_test_acc)
            test_acc_config1.append(max_test_acc)
        test_acc_config1 = np.array(test_acc_config1)
        print('Mean test accuracy [Config 1]:', np.mean(test_acc_config1))
        print('Std dev test accuracy [Config 1]:', np.std(test_acc_config1))
        results['config1']['graphs']=graphs
        results['config1']['test_acc_mean']=np.mean(test_acc_config1)
        results['config1']['test_acc_std']=np.std(test_acc_config1)

    #Setting 2
    configuration['batch_size']=256
    train_iterator = DataGenerator(
        train_dataset,
        shuffle=True,
        batch_size=configuration.get('batch_size'),
        num_workers=configuration.get('num_workers')
    )
    test_iterator = DataGenerator(
        test_dataset,
        shuffle=False,
        batch_size=configuration.get('batch_size'),
        num_workers=configuration.get('num_workers')
    )
    
    
    test_acc_config2=[]
    graphs=[]
    for i in range(5):
        network = ClassifierLN5(in_channels=1, number_of_classes=configuration.get('number_of_classes'), init_method='xavier_normal')
        model_configuration={
            'enable_log' : configuration.get('enable_log'),
            'model_path': pjoin(configuration.get('model_path'), dataset_name, 'config2'),
            'network' : network,
            'use_gpu': configuration.get('use_gpu'),
            'phases': ['train','test'],
            'loaders':{
                'train': train_iterator,
                'test':test_iterator
            },
            'loss': CELoss(),
            'learner': AdamOptimizer(network.parameters(), lr=1e-2, weight_decay=0),
            'scheduler': None,
            'max_epoch_val': configuration.get('max_epoch_val'),
            'initialization_method': configuration.get('initialization_method'),
            'acc_threshold': acc_threshold,
        }

        train_loss, train_acc, test_loss, test_acc = train(model_configuration)
        graphs.append({
            'train_loss':train_loss,
            'train_acc':train_acc,
            'test_loss':test_loss,
            'test_acc': test_acc
        })
        max_test_acc = max(test_acc)
        print('Max test accuracy [Config 2]:', max_test_acc)
        test_acc_config2.append(max_test_acc)
    test_acc_config2 = np.array(test_acc_config2)
    print('Mean test accuracy [Config 2]:', np.mean(test_acc_config2))
    print('Std dev test accuracy [Config 2]:', np.std(test_acc_config2))
    results['config2']['graphs']=graphs
    results['config2']['test_acc_mean']=np.mean(test_acc_config2)
    results['config2']['test_acc_std']=np.std(test_acc_config2)

    # Setting 3
    configuration['batch_size']=64
    train_iterator = DataGenerator(
        train_dataset,
        shuffle=True,
        batch_size=configuration.get('batch_size'),
        num_workers=configuration.get('num_workers')
    )
    test_iterator = DataGenerator(
        test_dataset,
        shuffle=False,
        batch_size=configuration.get('batch_size'),
        num_workers=configuration.get('num_workers')
    )

    test_acc_config3=[]
    graphs=[]
    for i in range(5):
        network = ClassifierLN5(in_channels=1, number_of_classes=configuration.get('number_of_classes'), init_method='he_normal')
        model_configuration={
            'enable_log' : configuration.get('enable_log'),
            'model_path': pjoin(configuration.get('model_path'), dataset_name, 'config3'),
            'network' : network,
            'use_gpu': configuration.get('use_gpu'),
            'phases': ['train','test'],
            'loaders':{
                'train': train_iterator,
                'test':test_iterator
            },
            'loss': CELoss(),
            'learner': AdamOptimizer(network.parameters(), lr=1e-3, weight_decay=0.01),
            'scheduler': None,
            'max_epoch_val': configuration.get('max_epoch_val'),
            'initialization_method': configuration.get('initialization_method'),
            'acc_threshold': acc_threshold,
        }

        train_loss, train_acc, test_loss, test_acc = train(model_configuration)
        graphs.append({
            'train_loss':train_loss,
            'train_acc':train_acc,
            'test_loss':test_loss,
            'test_acc': test_acc
        })
        max_test_acc = max(test_acc)
        print('Max test accuracy [Config 3]:', max_test_acc)
        test_acc_config3.append(max_test_acc)
    test_acc_config3 = np.array(test_acc_config3)
    print('Mean test accuracy [Config 3]:', np.mean(test_acc_config3))
    print('Std dev test accuracy [Config 3]:', np.std(test_acc_config3))
    results['config3']['graphs']=graphs
    results['config3']['test_acc_mean']=np.mean(test_acc_config3)
    results['config3']['test_acc_std']=np.std(test_acc_config3)

    #Setting 4
    configuration['batch_size']=64
    train_iterator = DataGenerator(
        train_dataset,
        shuffle=True,
        batch_size=configuration.get('batch_size'),
        num_workers=configuration.get('num_workers')
    )
    test_iterator = DataGenerator(
        test_dataset,
        shuffle=False,
        batch_size=configuration.get('batch_size'),
        num_workers=configuration.get('num_workers')
    )
    
    test_acc_config4=[]
    graphs=[]
    for i in range(5):
        network = ClassifierLN5(in_channels=1, number_of_classes=configuration.get('number_of_classes'), init_method='he_normal')
        model_configuration={
            'enable_log' : configuration.get('enable_log'),
            'model_path': pjoin(configuration.get('model_path'), dataset_name, 'config4'),
            'network' : network,
            'use_gpu': configuration.get('use_gpu'),
            'phases': ['train','test'],
            'loaders':{
                'train': train_iterator,
                'test':test_iterator
            },
            'loss': CELoss(),
            'learner': RMSOptimizer(network.parameters(), lr=1e-3, weight_decay=0.01),
            'scheduler': None,
            'max_epoch_val': configuration.get('max_epoch_val'),
            'initialization_method': configuration.get('initialization_method'),
            'acc_threshold': acc_threshold,
        }

        train_loss, train_acc, test_loss, test_acc = train(model_configuration)
        graphs.append({
            'train_loss':train_loss,
            'train_acc':train_acc,
            'test_loss':test_loss,
            'test_acc': test_acc
        })
        max_test_acc = max(test_acc)
        print('Max test accuracy [Config 4]:', max_test_acc)
        test_acc_config4.append(max_test_acc)
    test_acc_config4 = np.array(test_acc_config4)
    print('Mean test accuracy [Config 4]:', np.mean(test_acc_config4))
    print('Std dev test accuracy [Config 4]:', np.std(test_acc_config4))
    results['config4']['graphs']=graphs
    results['config4']['test_acc_mean']=np.mean(test_acc_config4)
    results['config4']['test_acc_std']=np.std(test_acc_config4)

    # Setting 5
    configuration['batch_size']=128
    train_iterator = DataGenerator(
        train_dataset,
        shuffle=True,
        batch_size=configuration.get('batch_size'),
        num_workers=configuration.get('num_workers')
    )
    test_iterator = DataGenerator(
        test_dataset,
        shuffle=False,
        batch_size=configuration.get('batch_size'),
        num_workers=configuration.get('num_workers')
    )

    test_acc_config5=[]
    graphs=[]
    for i in range(5):
        network = ClassifierLN5(in_channels=1, number_of_classes=configuration.get('number_of_classes'), init_method='he_uniform')
        model_configuration={
            'enable_log' : configuration.get('enable_log'),
            'model_path': pjoin(configuration.get('model_path'), dataset_name, 'config5'),
            'network' : network,
            'use_gpu': configuration.get('use_gpu'),
            'phases': ['train','test'],
            'loaders':{
                'train': train_iterator,
                'test':test_iterator
            },
            'loss': CELoss(),
            'learner': SGDOptimizer(network.parameters(), lr=1e-3, weight_decay=0.001),
            'scheduler': None,
            'max_epoch_val': configuration.get('max_epoch_val'),
            'initialization_method': configuration.get('initialization_method'),
            'acc_threshold': acc_threshold,
        }

        train_loss, train_acc, test_loss, test_acc = train(model_configuration)
        graphs.append({
            'train_loss':train_loss,
            'train_acc':train_acc,
            'test_loss':test_loss,
            'test_acc': test_acc
        })
        max_test_acc = max(test_acc)
        print('Max test accuracy [Config 5]:', max_test_acc)
        test_acc_config5.append(max_test_acc)
    test_acc_config5 = np.array(test_acc_config4)
    print('Mean test accuracy [Config 5]:', np.mean(test_acc_config5))
    print('Std dev test accuracy [Config 5]:', np.std(test_acc_config5))
    results['config5']['graphs']=graphs
    results['config5']['test_acc_mean']=np.mean(test_acc_config5)
    results['config5']['test_acc_std']=np.std(test_acc_config5)

    with open(dataset_name+'_results.json','w') as f:
        json.dump(results, f)

    drawGraphs1b(results, dataset_name)

#Driver routine for problem 1c
def problem1c(train_dataset, test_dataset, new_train_dataset, new_test_dataset):
    # lr = 1e-3, decay = 0.001, batch size = 64
    configuration['batch_size']=64
    train_iterator = DataGenerator(
        train_dataset,
        shuffle=True,
        batch_size=configuration.get('batch_size'),
        num_workers=configuration.get('num_workers')
    )
    test_iterator = DataGenerator(
        test_dataset,
        shuffle=False,
        batch_size=configuration.get('batch_size'),
        num_workers=configuration.get('num_workers')
    )

    new_train_iterator = DataGenerator(
        new_train_dataset,
        shuffle=True,
        batch_size=configuration.get('batch_size'),
        num_workers=configuration.get('num_workers')
    )
    new_test_iterator = DataGenerator(
        new_test_dataset,
        shuffle=False,
        batch_size=configuration.get('batch_size'),
        num_workers=configuration.get('num_workers')
    )
    
    # Train LeNet on positives and test on negatives
    network = ClassifierLN5(in_channels=1, number_of_classes=configuration.get('number_of_classes'), init_method='uniform')
    model_configuration={
        'enable_log' : configuration.get('enable_log'),
        'model_path': pjoin(configuration.get('model_path'), 'mnist_w_neg'),
        'network' : network,
        'use_gpu': configuration.get('use_gpu'),
        'phases': ['train','test'],
        'loaders':{
            'train': train_iterator,
            'test':test_iterator
        },
        'loss': CELoss(),
        'learner': AdamOptimizer(network.parameters(), lr=1e-3, weight_decay=0),
        'scheduler': None,
        'max_epoch_val': 50,
        'initialization_method': configuration.get('initialization_method'),
        'acc_threshold': 0.99,
    }

    train_loss, train_acc, test_loss, test_acc = train(model_configuration)

    lenet5 = [train_loss, train_acc, test_loss, test_acc]
    
    #Custom classifier training
    network = CustomClassifier()
    model_configuration={
        'enable_log' : configuration.get('enable_log'),
        'model_path': pjoin(configuration.get('model_path'), 'mnist_new', 'config3'),
        'network' : network,
        'use_gpu': configuration.get('use_gpu'),
        'phases': ['train','test'],
        'loaders':{
            'train': new_train_iterator,
            'test': new_test_iterator
        },
        'loss': CELoss(),
        'learner': AdamOptimizer(network.parameters(), lr=1e-3, weight_decay=0.001),
        'scheduler': None,
        'max_epoch_val': configuration.get('max_epoch_val'),
        'initialization_method': configuration.get('initialization_method'),
        'acc_threshold': 0.995
    }

    graphs = train(model_configuration)
    
    results = {
        'custom_classifier': graphs,
        'lenet': lenet5
    }

    with open('problem_1c.json', 'w') as f:
        f.write(json.dumps(results))

    drawGraphs1c(results)

#Helper function for saving results for problem 1c
def drawGraphs1c(results):
    for model in ['lenet', 'custom_classifier']:
        train_loss, train_acc, test_loss, test_acc = results[model]
        running_train_max_acc = [train_acc[np.argmax(test_acc)]]
        running_train_min_loss = [train_loss[np.argmax(test_acc)]]
        running_test_max_acc = [test_acc[np.argmax(test_acc)]]
        running_test_min_loss = [test_loss[np.argmax(test_acc)]]
        plt.figure()
        plt.plot(train_loss, label='train')
        plt.plot(test_loss, label='test')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Loss')
        plt.legend()
        plt.savefig(f'{model}_loss.png')
        plt.figure()
        plt.plot(train_acc, label='train')
        plt.plot(test_acc, label='test')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy')
        plt.legend()
        plt.savefig(f'{model}_train_acc.png')
        print(f'{model} Max Train Accuracy: {running_train_max_acc}')
        print(f'{model} Max Test Accuracy: {running_test_max_acc}')

#Computing histogram and save random samples
def compute_stats(mnist_train_set, mnist_test_set, mnist_train_neg_set, minst_test_neg_set):
    mnists_transform = BuildTransform([
        Rescale((28,28)),
        ConvertToTensor(),
        AdjustNormal(mean=[0.1307],
        std=[0.3081])
    ])
    mnist_pos_inv_trans = BuildTransform([
        AdjustNormal(mean=[-0.1307/0.3081],
                    std=[1/0.3081])
    ])
    mnists_neg_transform = BuildTransform([
        Rescale((28,28)),
        Reverse(),
        ConvertToTensor(),
        AdjustNormal(mean=[0.8691],
        std=[0.3018])
    ])
    mnist_neg_inv_trans = BuildTransform([
        AdjustNormal(mean=[-0.8691/0.3018],
                    std=[1/0.3018])
    ])

    train_ch = random.sample(range(0, len(mnist_train_dataset)), 10)
    neg_train_ch = random.sample(range(0, len(mnist_neg_train_dataset)), 10)
    
    test_ch = random.sample(range(0, len(mnist_test_dataset)), 10)
    f, ax = plt.subplots(4,10)
    for i in range(10):
        ax[0,i].imshow(mnist_train_dataset[train_ch[i]][0][0].cpu().numpy())
        ax[1,i].imshow(mnist_neg_train_dataset[neg_train_ch[i]][0][0].cpu().numpy())
        ax[2,i].imshow(mnist_test_dataset[test_ch[i]][0][0].cpu().numpy())
        ax[3,i].imshow(mnist_neg_test_dataset[test_ch[i]][0][0].cpu().numpy())
    
    plt.savefig('samples.png')

    nb_bins = 256
    count = np.zeros(nb_bins)
    print(len(mnist_train_dataset), len(mnist_test_dataset), len(mnist_neg_train_dataset), len(mnist_neg_test_dataset))
    for idx in range(len(mnist_train_dataset)):    
        x = mnist_pos_inv_trans(mnist_train_dataset[idx][0])*255
        x[x<1e-5]=0
        hist = np.histogram(x[0], bins=nb_bins, range=[0, 255])
        count += hist[0]
    
    for idx in range(len(mnist_test_dataset)):    
        x = mnist_pos_inv_trans(mnist_test_dataset[idx][0])*255
        x[x<1e-5]=0
        hist = np.histogram(x[0], bins=nb_bins, range=[0, 255])
        count += hist[0]
    bins = hist[1]
    fig = plt.figure()
    plt.bar(bins[:-1], count)
    plt.title('Positive MNIST Histogram')
    plt.savefig('positive_stats.png')
    
    neg_count = np.zeros(nb_bins)
    for idx in range(len(mnist_neg_train_dataset)):    
        x = mnist_neg_inv_trans(mnist_neg_train_dataset[idx][0])*255
        x[x<1e-5]=0
        hist = np.histogram(x[0], bins=nb_bins, range=[0, 255])
        neg_count += hist[0]
    
    for idx in range(len(mnist_neg_test_dataset)):    
        x = mnist_neg_inv_trans(mnist_neg_test_dataset[idx][0])*255
        x[x<1e-5]=0
        hist = np.histogram(x[0], bins=nb_bins, range=[0, 255])

        neg_count += hist[0]
        
    bins = hist[1]
    plt.figure()
    plt.bar(bins[:-1], neg_count)
    plt.title('Negative MNIST Histogram')
    plt.savefig('negative_stats.png')
    

if __name__=='__main__':

    mnists_transform = BuildTransform([
        Rescale((28,28)),
        ConvertToTensor(),
        AdjustNormal(mean=[0.1307],
        std=[0.3081])
    ])

    mnists_transform_alt = BuildTransform([
        Rescale((28,28)),
        ConvertToTensor(),
    ])

    cifar_transform = BuildTransform([
        Rescale((32,32)),
        ConvertToTensor(),
        AdjustNormal(mean=[0.486, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    ])

    mnists_neg_transform = BuildTransform([
        Rescale((28,28)),
        Reverse(),
        ConvertToTensor(),
        AdjustNormal(mean=[0.8691],
            std=[0.3018])
    ])

    mnists_neg_transform_alt = BuildTransform([
        Rescale((28,28)),
        Reverse(),
        ConvertToTensor(),
    ])

    mnist_pos_inv_trans = BuildTransform([
        AdjustNormal(mean=[-0.1307/0.3081],
                    std=[1/0.3081])
    ])

    mnist_neg_inv_trans = BuildTransform([
        AdjustNormal(mean=[-0.8691/0.3018],
                    std=[1/0.3018])
    ])

    mnist_train_dataset = MNISTDataset(download=True, root='data/mnist', train=True, transform=mnists_transform)
    mnist_test_dataset = MNISTDataset(download=True, root='data/mnist', train=False, transform=mnists_transform)

    mnist_neg_train_dataset = MNISTDataset(download=True, root='data/mnist', train=True, transform=mnists_neg_transform)
    mnist_neg_test_dataset = MNISTDataset(download=True, root='data/mnist', train=False, transform=mnists_neg_transform)

    fash_mnist_train_dataset = FashMNISTDataset(download=True, root='data/fashmnist', train=True, transform=mnists_transform)
    fash_mnist_test_dataset = FashMNISTDataset(download=True, root='data/fashmnist', train=False, transform=mnists_transform)
    
    cifar_train_dataset = CIFARDataset(download=True, root='data/cifar', train=True, transform=cifar_transform)
    cifar_test_dataset = CIFARDataset(download=True, root='data/cifar', train=False, transform=cifar_transform)
    
    # Problem 1b 
    #MNIST training 
    problem1b(mnist_train_dataset, mnist_test_dataset, 'mnist', 0.99)
    #Fashion MNIST training
    problem1b(fash_mnist_train_dataset, fash_mnist_test_dataset, 'fash_mnist', 0.9)
    #CIFAR training
    problem1b(cifar_train_dataset, cifar_test_dataset, 'cifar', 0.65)
    

    #Problem 1c

    mnist_train_dataset_alt = MNISTDataset(download=True, root='data/mnist', train=True, transform=mnists_transform_alt)
    mnist_test_dataset_alt = MNISTDataset(download=True, root='data/mnist', train=False, transform=mnists_transform_alt)

    mnist_neg_train_dataset_alt = MNISTDataset(download=True, root='data/mnist', train=True, transform=mnists_neg_transform_alt)
    mnist_neg_test_dataset_alt = MNISTDataset(download=True, root='data/mnist', train=False, transform=mnists_neg_transform_alt)

    # Computing mean and std for MNIST positives and negatives
    tmp_loader = torch.utils.data.DataLoader(mnist_train_dataset_alt,
                         batch_size=64,
                         num_workers=2,
                         shuffle=False)
    mean = 0.
    std = 0.
    for data, _ in tmp_loader:
        data = data.view(data.size(0), data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
    
    print('Positives Train(mean,std): ', mean/len(mnist_train_dataset_alt), std/len(mnist_train_dataset_alt))

    tmp_loader = torch.utils.data.DataLoader(mnist_test_dataset_alt,
                         batch_size=64,
                         num_workers=2,
                         shuffle=False)
    mean_acc = mean
    std_acc = std
    mean = 0.
    std = 0.
    for data, _ in tmp_loader:
        data = data.view(data.size(0), data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)

    mean_acc += mean
    std_acc += std
    print('Positives Test(mean,std): ', mean/len(mnist_test_dataset_alt), std/len(mnist_test_dataset_alt))
    print('Positives Overall(mean,std): ', mean_acc/(len(mnist_train_dataset_alt)+len(mnist_test_dataset_alt)), std_acc/(len(mnist_train_dataset_alt)+len(mnist_test_dataset_alt)))


    tmp_loader = torch.utils.data.DataLoader(mnist_neg_train_dataset_alt,
                         batch_size=64,
                         num_workers=2,
                         shuffle=False)
    mean = 0.
    std = 0.
    for data, _ in tmp_loader:
        data = data.view(data.size(0), data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
    
    print('Negatives Train(mean,std): ', mean/len(mnist_neg_train_dataset_alt), std/len(mnist_neg_train_dataset_alt))

    tmp_loader = torch.utils.data.DataLoader(mnist_neg_test_dataset_alt,
                         batch_size=10,
                         num_workers=0,
                         shuffle=False)
    mean_acc = mean
    std_acc = std
    mean = 0.
    std = 0.
    for data, _ in tmp_loader:
        data = data.view(data.size(0), data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
    mean_acc += mean
    std_acc += std
    print('Negatives Test(mean,std): ', mean/len(mnist_neg_test_dataset_alt), std/len(mnist_neg_test_dataset_alt))
    print('Negatives Overall(mean,std): ', mean_acc/(len(mnist_neg_train_dataset_alt)+len(mnist_neg_test_dataset_alt)), std_acc/(len(mnist_train_dataset_alt)+len(mnist_test_dataset_alt)))
    

    compute_stats(mnist_train_dataset, mnist_test_dataset, mnist_neg_train_dataset, mnist_neg_test_dataset)
    
    mnist_new_train_dataset = MergeDatasets([mnist_train_dataset, mnist_neg_train_dataset])
    mnist_new_test_dataset = MergeDatasets([mnist_test_dataset, mnist_neg_test_dataset])

    # Call to problem1c driver
    problem1c(mnist_train_dataset, mnist_neg_test_dataset, mnist_new_train_dataset, mnist_new_test_dataset)
