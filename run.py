import os
import argparse
import time
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
import random
import numpy as np
import csv
from datetime import datetime


script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print("Current working directory:", os.getcwd())


def get_args():
    parser = argparse.ArgumentParser(description='SeesawNet')

    # basic config
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
    parser.add_argument('--model', type=str, required=False, default='SeesawNet', help='model name')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    
    # data loader
    parser.add_argument('--data', type=str, required=False, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=720, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--enc_in', type=int, default=7, help='channel_decoder input size')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    
    # model define
    parser.add_argument('--pd_layers', type=int, default=1, help='num of patch dependency attention layers')
    parser.add_argument('--cr_layers', type=int, default=1, help='num of channel relationship attention layers')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
    parser.add_argument('--down_sample_rate', type=float, default=0.5, help='down sample rate for aggregation layer')

    # optional parameters
    parser.add_argument('--group', type=bool, default=False, help="whether to use group attention. If token number is large, group attention can reduce memory usage.")
    parser.add_argument('--patch_len', type=int, default=24, help='length of patches')
    parser.add_argument('--stride', type=int, default=24, help='stride')
    parser.add_argument('--norm', type=str, default="BatchNorm", help='Normalization, [LayerNorm, BatchNorm]')
    parser.add_argument('--pe', type=str, default='zeros', help='positional encoding type')
    parser.add_argument('--learn_pe', type=bool, default=True, help='whether to learn positional encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--alpha', type=float, default=0.35, help='weight of time-frequency MAE loss')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function, [MSE, TFMAE, NDLoss]')

    # optimization
    parser.add_argument('--itr', type=int, default=3, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.1, help='optimizer learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
    
    args = parser.parse_args()

    # GPU adjustment
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids[:torch.cuda.device_count()]]
        args.gpu = args.device_ids[0]
    
    return args

if __name__ == '__main__':
    args = get_args()
    args.ii = 0
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.record = f"""
    \tExp   \t model_id: {args.model_id}, model: {args.model}, time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    \tTask  \t data: {args.data_path.split('.')[0]}, seq_len: {args.seq_len}, pred_len: {args.pred_len}, channels: {args.enc_in}
    \tModel \t pd_layers: {args.pd_layers}, cr_layers: {args.cr_layers}, d_model: {args.d_model}, d_ff: {args.d_ff}, n_heads: {args.n_heads}, dropout: {args.dropout}
    \tOptim \t batch_size: {args.batch_size}, learning_rate: {args.learning_rate}, lradj: {args.lradj}, pct_start: {args.pct_start}
    \tTrain \t is_training: {args.is_training}, itr: {args.itr}, train_epochs: {args.train_epochs}, patience: {args.patience}
    """
    print(f'Args in experiment:{args.record}')


    Exp = Exp_Long_Term_Forecast

    mses, maes, stop_epochs = [], [], []

    if args.is_training:
        for ii in range(args.itr):
            args.ii = ii
            # setting record of experiments
            setting = '{}_{}_{}_seq{}_pred{}_bs{}_lr{}_pd{}_cr{}_dm{}_df{}_nh{}_dp{}_dsr{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data_path.split('.')[0],
                args.seq_len,
                args.pred_len,
                args.batch_size,
                args.learning_rate,
                args.pd_layers,
                args.cr_layers,
                args.d_model,
                args.d_ff,
                args.n_heads,
                args.dropout,
                args.down_sample_rate,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            stop_epoch = exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            mse, mae = exp.test(setting)

            mses.append(mse)
            maes.append(mae)
            stop_epochs.append(stop_epoch)
            print('iter : {} / {} | StopEpoch: {} | mse: {:.4f}, mae: {:.4f}\n'.format(
                ii+1, args.itr, stop_epoch, mse, mae))

            torch.cuda.empty_cache()
    
        mse_avg, mae_avg = np.mean(mses), np.mean(maes)
        mse_std, mae_std = np.std(mses), np.std(maes)
        result_path = './csv_results/' + args.model + '/' + args.model_id + '/'
        csv_name = result_path + 'results.csv'
        if not os.path.exists(csv_name):
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            with open(csv_name, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                header = ['Model', 'Dataset', 'SeqLen', 'PredLen', 'Itr', 'Loss', 
                          'MSE', 'MAE', 'MSE_std', 'MAE_std', 'Time', 'Params']
                csvwriter.writerow(header)
        res = [args.model_id, args.data_path.split('.')[0], args.seq_len, args.pred_len, 'avg', args.loss, 
                mse_avg, mae_avg, mse_std, mae_std, datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), args]
        with open(csv_name, 'a+', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(res)
        print('Args Setting:', args.record)
        print('MSE  All: {} | Avg: {:.4f} ± {:.4f}'.format(mses, mse_avg, mse_std))
        print('MAE  All: {} | Avg: {:.4f} ± {:.4f}'.format(maes, mae_avg, mae_std))
    
    else:
        ii = 0
        setting = '{}_{}_{}_seq{}_pred{}_bs{}_lr{}_pd{}_cr{}_dm{}_df{}_nh{}_dp{}_dsr{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data_path.split('.')[0],
                args.seq_len,
                args.pred_len,
                args.batch_size,
                args.learning_rate,
                args.pd_layers,
                args.cr_layers,
                args.d_model,
                args.d_ff,
                args.n_heads,
                args.dropout,
                args.down_sample_rate,
                args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        start_time = time.time()
        print(exp.test(setting, test=1))
        end_time = time.time()
        print(f"Runtime: {end_time - start_time:.4f} seconds")
        torch.cuda.empty_cache()
