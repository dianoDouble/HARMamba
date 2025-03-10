
import os
import json

import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader



def build_hardataset(data_path,is_train, args):
    if is_train :
        '''Dataset loading'''
        print('==> Preparing data..')
        train_x = np.load(data_path + '/x_train.npy')
        train_x = torch.from_numpy(train_x).float()
        train_x = train_x.unsqueeze(1)
        train_y = (np.load(data_path + '/y_train.npy'))

        train_y = np.argmax(train_y, axis=1)
        train_y = torch.from_numpy(train_y).float()
        train_y = train_y.unsqueeze(1)
        test_x = np.load(data_path + '/x_test.npy')
        test_x = torch.from_numpy(test_x).float()
        test_x = test_x.unsqueeze(1)
        test_x = test_x.type(torch.FloatTensor)
        test_y = np.load(data_path + '/y_test.npy')
        test_y = np.argmax(test_y, axis=1)
        test_y = torch.from_numpy(test_y).float()
        test_y = test_y.unsqueeze(1)
        test_y = test_y.type(torch.FloatTensor)
        category = len(np.unique(train_y))
        print('\n==================================================  【张量转换】  ===================================================\n')
        print('x_train_tensor shape: %s\nx_test_tensor shape: %s' % (train_x.shape, test_x.shape))
        print('y_train_tensor shape: %s\ny_test_tensor shape: %s' % (train_y.shape, test_y.shape))
        print('Category num: %d' % (category))
        train_data = TensorDataset(train_x, train_y)
        # print('train_data:'+ str(train_data.shape))
        train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers,
                                 pin_memory=args.pin_mem, drop_last=True)
        return train_loader, category
    else :
        test_x = np.load(data_path + '/x_test.npy')
        test_x = torch.from_numpy(test_x).float()
        test_x = test_x.unsqueeze(1)
        test_x = test_x.type(torch.FloatTensor)
        test_y = np.load(data_path + '/y_test.npy')
        test_y = np.argmax(test_y, axis=1)
        test_y = torch.from_numpy(test_y).float()
        test_y = test_y.unsqueeze(1)
        test_y = test_y.type(torch.FloatTensor)
        category = len(np.unique(test_y))
        test_data = TensorDataset(test_x, test_y)
        # print('test_data:' + str(test_data.shape))

        test_loader = DataLoader(test_data,batch_size=args.batch_size,num_workers=args.num_workers,
                                 pin_memory=args.pin_mem,drop_last=False )
        return test_loader, category

def build_dataset(is_train, args):
    # transform = build_transform(is_train, args)
    if args.data_set == 'WISDM' or args.data_set == 'uci' or args.data_set == 'pamap2' or args.data_set == 'unimib':
        dataset, nb_classes = build_hardataset(args.data_path, is_train = is_train, args=args)


    return dataset, nb_classes



