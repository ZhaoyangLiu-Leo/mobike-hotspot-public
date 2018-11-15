# -*- coding: utf-8 -*-

import sys
import os
import argparse

sys.path.insert(0, './')
sys.path.insert(0, '../')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from models.dann_ranking import run_rank_reg_one_time, grid_search_rank_reg
from models.dann_ranking_cnn import run_one_time_cnn, grid_search_cnn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dann ranking model")
    parser.add_argument('--model', help='model choice for feature network component, mlp/cnn', required=True,
                        choices=['mlp', 'cnn'], default='mlp')
    parser.add_argument('--train_mode',
                        help='whether combine the domain adaptation process, training mode: source / dann',
                        required=True, choices=['source', 'dann'], default='source', type=str)
    parser.add_argument('--gpu', help='select gpu device', required=False, default=0, type=int)
    parser.add_argument(
        '--grid_search', help='whether grid search for hyper parameters', action='store_true')
    parser.add_argument('--search_mode', help='grid search / random search', choices=['grid', 'random'], required=False,
                        default='grid', type=str)
    parser.add_argument('--source_city', help='source city to transfer',
                        required=True, choices=['bj', 'sh', 'nb'], type=str)
    parser.add_argument('--target_city', help='target city to predict',
                        required=True, choices=['bj', 'sh', 'nb'], type=str)
    parser.add_argument('--task_type', required=False, default='reg', choices=['reg', 'cls'], type=str,
                        help='regression task or classification task')
    parser.add_argument('--hot_count', required=True, default=100, type=int,
                        help='hotspot count')
    parser.add_argument('--max_train_steps', required=False, default=10001, type=int,
                        help='max train num steps')
    parser.add_argument('--repeat', required=False, default=1, type=int, help='repeat time for grid search')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.model == 'mlp':
        if args.grid_search:
            grid_search_rank_reg(train_mode=args.train_mode, search_mode=args.search_mode, repeat=args.repeat,
                                 city_pair=(args.source_city, args.target_city),
                                 data_version='bound', task_type=args.task_type,
                                 hot_count=args.hot_count, max_train_steps=args.max_train_steps)
        else:
            run_rank_reg_one_time(
                train_mode=args.train_mode,
                city_pair=(args.source_city, args.target_city),
                path_pattern='../data/road/train_bound/%s_500_week.csv',
                task_type=args.task_type, hot_count=args.hot_count,
                max_train_steps=args.max_train_steps
            )
    elif args.model == 'cnn':
        if args.grid_search:
            grid_search_cnn(train_mode=args.train_mode, search_mode=args.search_mode, repeat=args.repeat,
                            city_pair=(args.source_city, args.target_city),
                            data_version='bound_cnn', task_type=args.task_type,
                            hot_count=args.hot_count, max_train_steps=args.max_train_steps)
        else:
            run_one_time_cnn(
                train_mode=args.train_mode,
                city_pair=(args.source_city, args.target_city),
                path_pattern='../data/road/train_bound_cnn/%s_500_week.csv',
                task_type=args.task_type, hot_count=args.hot_count,
                max_train_steps=args.max_train_steps
            )
    else:
        pass
