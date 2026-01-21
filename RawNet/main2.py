import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import yaml
from du2 import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval,Dataset_ASVspoof2021_train
#from model import RawNet
from rawnet1 import RawNet
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed

__author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"
__credits__ = ["Jose Patino", "Massimiliano Todisco", "Jee-weon Jung"]


def evaluate_accuracy(dev_loader, model, device):
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    for batch_x, batch_y in dev_loader:
        
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        # 修改：处理可能的双返回值
        batch_out = model(batch_x)
        if isinstance(batch_out, tuple):
            batch_out = batch_out[0]  # 只取分类输出
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
    return 100 * (num_correct / num_total)

def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=30, shuffle=False, drop_last=False)
    model.eval()
    
    for batch_x,utt_id in data_loader:
        fname_list = []
        score_list = []  
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        batch_out = model.predict_with_uq(batch_x)
        batch_score = (batch_out[:, 1]
                       ).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        
        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list,score_list):
                fh.write('{} {}\n'.format(f, cm))
        fh.close()   
    print('Scores saved to {}'.format(save_path))

def train_epoch(train_loader, model, lr,optim, device):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii = 0
    model.train()

    #set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    for batch_x, batch_y in train_loader:
       
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)
        if ii % 10 == 0:
            sys.stdout.write('\r \t {:.2f}'.format(
                (num_correct/num_total)*100))
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
       
    running_loss /= num_total
    train_accuracy = (num_correct/num_total)*100
    return running_loss, train_accuracy
    
#def produce_evaluation_file(dataset, model, device, save_path, uncertainty_path=None):
#    data_loader = DataLoader(dataset, batch_size=50, shuffle=False, drop_last=False)
#    model.eval()
#    
#    # 确保不确定性输出
##    if uncertainty_path is None and hasattr(model, 'delta_uq_enabled') and model.delta_uq_enabled:
##        uncertainty_path = save_path.replace('.log', '_uncertainty.log')
#    
##    if uncertainty_path:
##        uncertainty_file = open(uncertainty_path, 'w')
#    
#    for batch_x, utt_id in data_loader:
#        fname_list = []
#        score_list = []  
#        uncertainty_list = []
#        
#        batch_size = batch_x.size(0)
#        batch_x = batch_x.to(device)
#        
##        if hasattr(model, 'delta_uq_enabled') and model.delta_uq_enabled:
##            model_output = model(batch_x, return_uncertainty=True)
##            if isinstance(model_output, tuple):
##                logits, uncertainty = model_output
##            else:
##                logits = model_output
##                uncertainty = None
##		    
##		    # 正常计算 softmax
##            softmax_pred = torch.nn.functional.softmax(logits, dim=1)
##            batch_score = softmax_pred[:, 1].data.cpu().numpy().ravel()
##		
##		    # 保存不确定性
##            if uncertainty is not None:
##                uncertainty_np = uncertainty.squeeze().data.cpu().numpy()
##                if uncertainty_np.ndim == 0:
##                    uncertainty_list.extend([uncertainty_np] * batch_size)
##                else:
##                    uncertainty_list.extend(uncertainty_np.tolist())
##        else:
#        logits = model(batch_x)
#        softmax_pred = torch.nn.functional.softmax(logits, dim=1)
#        batch_score = softmax_pred[:, 1].data.cpu().numpy().ravel()
#        
#        
#        fname_list.extend(utt_id)
#        score_list.extend(batch_score.tolist())
#        
#        # 保存结果
#        with open(save_path, 'a+') as fh:
#            for f, cm in zip(fname_list, score_list):
#                fh.write('{} {}\n'.format(f, cm))
#        
#        # 保存不确定性
##        if uncertainty_path and uncertainty_list:
##            for f, unc in zip(fname_list, uncertainty_list):
##                uncertainty_file.write('{} {}\n'.format(f, unc))
#    
##    if uncertainty_path:
##        uncertainty_file.close()
##        print('Uncertainty scores saved to {}'.format(uncertainty_path))
#    
#    print('Scores saved to {}'.format(save_path))
#      
#
#def train_epoch(train_loader, model, lr, optim, device):
#    running_loss = 0
#    num_correct = 0.0
#    num_total = 0.0
#    ii = 0
#    model.train()
#
#    weight = torch.FloatTensor([0.1, 0.9]).to(device)
#    criterion = nn.CrossEntropyLoss(weight=weight)
#    
#    for batch_x, batch_y in train_loader:
#        batch_size = batch_x.size(0)
#        num_total += batch_size
#        ii += 1
#        batch_x = batch_x.to(device)
#        batch_y = batch_y.view(-1).type(torch.int64).to(device)
#        
#        # 统一训练接口
##        if hasattr(model, 'delta_uq_enabled') and model.delta_uq_enabled:
##            # 训练时计算不确定性
##            model_output = model(batch_x, train=True, return_uncertainty=True)
##            
##            # 正确解包输出
##            if isinstance(model_output, tuple):
##                output, uncertainty = model_output
##            else:
##                output = model_output
##                uncertainty = None
#            
#            # 关键修复：确保输出形状正确
##            if output.dim() == 3:
##                print(f"Warning: Output is 3D, taking last time step. Shape: {output.shape}")
##                output = output[:, -1, :]  # 取最后一个时间步
##            
##            if output.dim() != 2 or output.size(1) != 2:
##                print(f"Error: Unexpected output shape: {output.shape}")
##                # 如果输出形状不对，重新计算（不使用不确定性）
##                output = model(batch_x)
##                if output.dim() == 3:
##                    output = output[:, -1, :]
##                uncertainty = None
#            
#            # 基础分类损失
##            cls_loss = criterion(output, batch_y)
##            
##            # 不确定性正则化
##            if uncertainty is not None:
##                uncertainty_reg = torch.mean(uncertainty)
##                # 组合损失
##                total_loss = cls_loss + model.uncertainty_weight * uncertainty_reg
##            else:
##                total_loss = cls_loss
#            
##        else:
#        output = model(batch_x)
#            # 确保输出形状正确
#        if output.dim() == 3:
#            output = output[:, -1, :]
#        total_loss = criterion(output, batch_y)
#        
#        # 计算准确率
#        with torch.no_grad():
#            if output.dim() == 2 and output.size(1) == 2:
#                _, batch_pred = output.max(dim=1)
#                num_correct += (batch_pred == batch_y).sum(dim=0).item()
#            else:
#                print(f"Warning: Cannot compute accuracy, output shape: {output.shape}")
#        
#        running_loss += total_loss.item() * batch_size
#        
#        if ii % 10 == 0:
#            sys.stdout.write('\r \t {:.2f}'.format((num_correct/num_total)*100))
#        
#        optim.zero_grad()
#        total_loss.backward()
#        optim.step()
#       
#    running_loss /= num_total
#    train_accuracy = (num_correct/num_total)*100
#    return running_loss, train_accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
    # Dataset
#    parser.add_argument('--database_path', type=str, default='/data/pytorch_lightning_FAD-main/data/ASVspoof2019_LA/', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 LA eval data folders are in the same database_path directory.')
#    '''
#    % database_path/
#    %   |- LA
#    %      |- ASVspoof2021_LA_eval/flac
#    %      |- ASVspoof2019_LA_train/flac
#    %      |- ASVspoof2019_LA_dev/flac
#    '''
#
#    parser.add_argument('--protocols_path', type=str, default='/data/pytorch_lightning_FAD-main/data/ASVspoof2019_LA/', help='Change with path to user\'s LA database protocols directory address')
#    '''
#    % protocols_path/
#    %   |- ASVspoof_LA_cm_protocols
#    %      |- ASVspoof2021.LA.cm.eval.trl.txt
#    %      |- ASVspoof2019.LA.cm.dev.trl.txt 
#    %      |- ASVspoof2019.LA.cm.train.trn.txt 
#    '''

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    # model
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')

    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--is_eval', action='store_true', default=False,help='eval database')
    parser.add_argument('--eval_part', type=int, default=0)
    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True, 
                        help='use cudnn-deterministic? (default true)')    
    
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False, 
                        help='use cudnn-benchmark? (default false)') 



    # 修改：添加不确定性输出参数
#    parser.add_argument('--uncertainty_output', type=str, default=None,
#                        help='Path to save the uncertainty scores')
    # DeltaUQ参数
    parser.add_argument('--delta_uq', action='store_true', default=True,
                        help='Use DeltaUQ for uncertainty quantification')
    # Auxiliary arguments
    parser.add_argument('--track', type=str, default='DF21',choices=['LA21', 'ITW','DF21'], help='LA/PA/DF')    ##===================================================Rawboost data augmentation ======================================================================#
    parser.add_argument('--algo', type=int, default=3, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')
    parser.add_argument('--da_prob', type=float, default=2)

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')
    
    ##===================================================Rawboost data augmentation ======================================================================#

    
    dir_yaml = os.path.splitext('model_config_RawNet')[0] + '.yaml'

    with open(dir_yaml, 'r') as f_yaml:
            parser1 = yaml.safe_load(f_yaml)

    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()
 
    #make experiment reproducible
    set_random_seed(args.seed, args)
    
     # 修改：设置DeltaUQ配置
	# 在模型配置中添加
    if args.delta_uq:
        parser1['model']['delta_uq'] = True 
#        parser1['model']['uncertainty_weight'] = 0.1  # 不确定性损失权重
#        print("Using DeltaUQ for uncertainty quantification with weight 0.1")
#    else:
#        parser1['model']['use_deltaUQ'] = False
#        print("Not using DeltaUQ")   
    track = args.track

#    assert track in ['LA', 'PA','DF'], 'Invalid track given'

    #database
#    prefix      = 'ASVspoof_{}'.format(track)
#    prefix_2019 = 'ASVspoof2019.{}'.format(track)
#    prefix_2021 = 'ASVspoof2021.{}'.format(track)
    
    #define model saving path
    model_tag = 'model_{}'.format(
        track)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)
    # 修改：在模型标签中包含DeltaUQ信息
    if args.delta_uq:
        model_tag = model_tag + '_75'
    model_save_path = os.path.join('models', model_tag)
    
    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    #model 
    model = RawNet(parser1['model'], device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model =(model).to(device)
    
    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path,map_location=device))
        print('Model loaded : {}'.format(args.model_path))

    # 数据集配置
    dataset_configs = {
        'LA21': {
            'protocols_dir': '/data/pytorch_lightning_FAD-main/data/protocols/ASVspoof2021_LA/',
            'data_base_dir': '/data/pytorch_lightning_FAD-main/data/ASVspoof2021_LA_eval/' ,
            'train_file': 'asvspoof2021_la.train.trn.txt',
            'dev_file': 'asvspoof2021_la.dev.trl.txt', 
            'eval_file': 'asvspoof2021_la.eval.trl.txt'
        },
        'DF21': {
            'protocols_dir': '/data/pytorch_lightning_FAD-main/data/protocols/ASVspoof2021_DF/',
            'data_base_dir': '/data/pytorch_lightning_FAD-main/data/aasist/datasets/ASVspoof2021_DF_eval/',
            'train_file': 'asvspoof2021_df.train.trn.txt',
            'dev_file': 'asvspoof2021_df.dev.trl.txt',
            'eval_file': 'asvspoof2021_df.eval.trl.txt'
        },
        'ITW': {
            'protocols_dir': '/data/pytorch_lightning_FAD-main/data/protocols/in_the_wild/',
            'data_base_dir': '/data/pytorch_lightning_FAD-main/data/aasist/datasets/release_in_the_wild/', 
            'train_file': 'in_the_wild.train.trn.txt',
            'dev_file': 'in_the_wild.dev.trl.txt',
            'eval_file': 'in_the_wild.eval.trl.txt'
        }
    }
    
    config = dataset_configs[args.track]
   
    #evaluation 
    if args.eval:
        # 评估三个数据集
#        datasets_to_eval = ['LA21', 'DF21', 'ITW']
        
#        for track in datasets_to_eval:
        print(f'\nEvaluating {track} dataset...')
        config = dataset_configs[track]
       
        file_eval = genSpoof_list(
           dir_meta=os.path.join(config['protocols_dir'], config['eval_file']),
           is_train=False, is_eval=True
       )
        print(f'No. of {track} eval trials: {len(file_eval)}')
       
        eval_set = Dataset_ASVspoof2021_eval(
           args, list_IDs=file_eval,
           base_dir=config['data_base_dir'], dataset_type=track
       )
       
       # 生成输出文件名
        output_file = f'std_{track}_da3_32.log'
#        uncertainty_file = f'uncertainty_{track}.log' if args.uncertainty_output else None
       
        produce_evaluation_file(
           eval_set, model, device, output_file
       )

        sys.exit(0)

     
    # define train dataloader
   
   # 加载训练数据
    d_label_trn, file_train = genSpoof_list(
       dir_meta=os.path.join(config['protocols_dir'], config['train_file']),
       is_train=True, is_eval=False
   )
    print(f'No. of {args.track} training trials: {len(file_train)}')
   
    train_set = Dataset_ASVspoof2021_train(
       args, list_IDs=file_train, labels=d_label_trn,
       base_dir=config['data_base_dir'], dataset_type=args.track
   )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
   
   # 加载验证数据
    d_label_dev, file_dev = genSpoof_list(
       dir_meta=os.path.join(config['protocols_dir'], config['dev_file']),
       is_train=False, is_eval=False
   )
    print(f'No. of {args.track} validation trials: {len(file_dev)}')
   
    dev_set = Dataset_ASVspoof2021_train(
       args, list_IDs=file_dev, labels=d_label_dev,
       base_dir=config['data_base_dir'], dataset_type=args.track
   )
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False)
    del dev_set,d_label_dev

    # Training and validation 
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    best_acc = 90
    for epoch in range(num_epochs):
        running_loss, train_accuracy = train_epoch(train_loader,model, args.lr,optimizer, device)
        valid_accuracy = evaluate_accuracy(dev_loader, model, device)
        writer.add_scalar('train_accuracy', train_accuracy, epoch)
        writer.add_scalar('valid_accuracy', valid_accuracy, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        print('\n{} - {} - {:.2f} - {:.2f}'.format(epoch,
                                                   running_loss, train_accuracy, valid_accuracy))
        
        if valid_accuracy > best_acc:
            print('best model find at epoch', epoch)
            torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))
        best_acc = max(valid_accuracy, best_acc)