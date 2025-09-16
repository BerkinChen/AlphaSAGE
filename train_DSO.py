import torch
import sklearn
import tensorflow as tf
import numpy as np
import os,json
import argparse
from datetime import datetime

from alphagen.data.expression import *
# from alphagen_qlib.calculator import QLibStockDataCalculator
from dso import DeepSymbolicRegressor
from dso.library import Token, HardCodedConstant
from dso import functions
from alphagen.models.alpha_pool import AlphaPool
from alphagen.utils import reseed_everything
from alphagen_generic.operators import funcs as generic_funcs
from alphagen_generic.features import *
from gan.utils.data import get_data_by_year

QLIB_PATH = '/DATA1/home/chenbq/AlphaStruct/data/qlib_data/cn_data_rolling'

funcs = {func.name: Token(complexity=1, **func._asdict()) for func in generic_funcs}
for i, feature in enumerate(['open', 'close', 'high', 'low', 'volume', 'vwap']):
    funcs[f'x{i+1}'] = Token(name=feature, arity=0, complexity=1, function=None, input_var=i)
for v in [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]:
    funcs[f'Constant({v})'] = HardCodedConstant(name=f'Constant({v})', value=v)

def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    tf.set_random_seed(args.seed)
    reseed_everything(args.seed)

    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    train_start_time = '2010-01-01'
    train_end_time = f'{args.train_end_year}-12-31'
    valid_start_time = f'{args.train_end_year + 1}-01-01'
    valid_end_time = f'{args.train_end_year + 1}-12-31'
    test_start_time = f'{args.train_end_year + 2}-01-01'
    test_end_time = f'{args.train_end_year + 4}-12-31'

    data = StockData(instrument=args.instruments,
                           start_time=train_start_time,
                           end_time=train_end_time,
                           qlib_path=QLIB_PATH)
    data_valid = StockData(instrument=args.instruments,
                           start_time=valid_start_time,
                           end_time=valid_end_time,
                           qlib_path=QLIB_PATH)
    data_test = StockData(instrument=args.instruments,
                          start_time=test_start_time,
                          end_time=test_end_time,
                          qlib_path=QLIB_PATH)


    cache = {}
    device = torch.device('cuda:0')

    X = np.array([['open_', 'close', 'high', 'low', 'volume', 'vwap']])
    y = np.array([[1]])
    functions.function_map = funcs

    pool = AlphaPool(capacity=args.pool,
                    stock_data=data,
                    target=target,
                    ic_lower_bound=None)
    save_path = f'out_dso/{args.name}_{args.instruments}_{args.pool}_{args.train_end_year}_{args.seed}/'
    os.makedirs(save_path,exist_ok=True)

    class Ev:
        def __init__(self, pool):
            self.cnt = 0
            self.pool = pool
            self.results = {}

        def alpha_ev_fn(self, key):
            expr = eval(key)
            try:
                ret = self.pool.try_new_expr(expr)
            except OutOfDataRangeError:
                ret = -1.
            else:
                ret = -1.
            finally:
                self.cnt += 1
                if self.cnt % 100 == 0:
                    test_ic = pool.test_ensemble(data_test,target)[0]
                    self.results[self.cnt] = test_ic
                    print(self.cnt, test_ic)
                return ret

    ev = Ev(pool)

    config = dict(
        task=dict(
            task_type='regression',
            function_set=list(funcs.keys()),
            metric='alphagen',
            metric_params=[lambda key: ev.alpha_ev_fn(key)],
        ),
        training={'n_samples': 5000, 'batch_size': 128, 'epsilon': 0.05},
        prior={'length': {'min_': 2, 'max_': 20, 'on': True}},
        experiment={'seed':args.seed},
    )

    # Create the model
    model = DeepSymbolicRegressor(config=config)
    model.fit(X, y)
    with open(f'{save_path}/pool.json', 'w') as f:
        json.dump(pool.to_dict(), f)
    print(ev.results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruments', type=str, default='csi300')
    parser.add_argument('--train-end-year', type=int, default=2018)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pool', type=int, default=100)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--name', type=str, default='test')
    args = parser.parse_args()
    run(args)