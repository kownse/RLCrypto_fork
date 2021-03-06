# -*- coding:utf-8 -*-
import sys
from utils.TradingUtils import *
from trader import *
from utils import config
from sklearn.preprocessing import StandardScaler
from models.ModelTrainer import *
import importlib
import torch.multiprocessing as mp

from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

import time
start_time = time.time()

CONFIG_PATH = './config/config.json'
if not os.path.exists(CONFIG_PATH):
    print("config file doesn't exist")
    sys.exit(1)
config.init_config(CONFIG_PATH)

init_account(config.account_file)
print(get_accounts())


class PortfolioManager(object):
    def __init__(self):
        self.portfolio = []
        self.asset_data = None
        self.all_asset_data = {}
        self.agent = None
        self.trader = None
    
    def init_assets(self, assets_config):
        if not os.path.exists(assets_config):
            print('Portfolio config file does not exist, run PortfolioManager first')
            return
        with open(assets_config, 'r') as f:
            print('Load portfolio successfully')
            self.portfolio = json.loads(f.read())
    
    def init_data(self, portfolio, bar_count, mode="huobi", data_interval=config.tick_interval):
        if len(portfolio) == 0:
            print('Load portfolio first')
            return

        self.data_mode = mode
        self.data_interval = data_interval

        if mode == "huobi":
            original_data = klines(portfolio, base_currency=config.base_currency, interval=data_interval, count=bar_count)
        elif mode == "local":
            original_data = klines_local(portfolio, interval=data_interval, begin_date=config.begin_date)

        self.asset_data = default_pre_process(original_data).fillna(0)
        return self.asset_data

    def check_init_data(self, portfolio, bar_count, mode="local", tick_interval=config.tick_interval):
        if tick_interval in self.all_asset_data:
            return
        print('load data for', portfolio)
        self.all_asset_data[tick_interval] = self.init_data(portfolio, bar_count, mode, tick_interval)
    
    def init_trader(self):
        self.trader = Trader(assets=self.portfolio,
                             base_currency=config.base_currency,
                             max_asset_percent=config.max_asset_percent,
                             max_order_waiting_time=config.max_order_waiting_time,
                             price_discount=config.price_discount,
                             amount_discount=config.amount_discount,
                             order_type=config.order_type,
                             trace_order=config.trace_order,
                             debug_mode=config.debug_mode)
    
    def load_model(self, path=config.model_path):
        if len(self.portfolio) == 0 or self.asset_data is None:
            print('Init data first')
            return

        self.agent = config.agent(s_dim=self.asset_data.shape[-1],
                                  b_dim=self.asset_data.shape[0],
                                  a_dim=2,
                                  learning_rate=config.learning_rate,
                                  batch_length=config.batch_length,
                                  rnn_layers=config.rnn_layers,
                                  normalize_length=config.normalize_length,
                                  rnn_type=config.rnn_type,
                                  drop=config.drop)
        self.agent.load_model(model_path=path)
    
    def build_model(self):
        if len(self.portfolio) == 0 or self.asset_data is None:
            print('Init data first')
            return
        self.agent = ModelTrainer.create_new_model(model_type=config.agent,
                                            asset_data=self.asset_data,
                                            c=config.fee,
                                            normalize_length=config.normalize_length,
                                            rnn_layers=config.rnn_layers,
                                            rnn_type=config.rnn_type,
                                            batch_length=config.batch_length,
                                            train_length=config.train_length,
                                            max_epoch=config.max_training_epoch,
                                            learning_rate=config.learning_rate,
                                            model_path=config.model_path,
                                            patient=config.patient,
                                            patient_rounds=config.patient_rounds)

    def build_model_batch(self):
        params = [
            ("DRL_Torch", 1, 'gru', 128, 0.2, 720, '1h', 55400, None, 'adam', ["data/USD_1h/BTC-USD_2013-03-31.csv"]),
            ("DRL_Torch", 1, 'gru', 128, 0.2, 720, '1h', 55400, None, 'adamax', ["data/USD_1h/BTC-USD_2013-03-31.csv"]),
            # ("DRL_Torch", 1, 'gru', 128, 0.2, 720, '1h', 55400, 'reduce', 'adam', ["data/USD_1h/BTC-USD_2013-03-31.csv"]),
            ("DRL_Torch", 1, 'gru', 128, 0.2, 720, '1h', 55400, 'cos', 'adam', ["data/USD_1h/BTC-USD_2013-03-31.csv"]),
            ("DRL_Torch", 1, 'gru', 128, 0.2, 720, '1h', 55400, 'cos', 'adamax', ["data/USD_1h/BTC-USD_2013-03-31.csv"]),
        ]

        processes = []

        for (model_type, rnn_layers, rnn_type, linear_base, drop, normalize_length, interval, train_length, lr_type, opt_type, portfolio) in params:
            self.check_init_data(portfolio, config.train_bar_count, mode='local', tick_interval=interval)
            
            # ModelTrainer.create_new_model(model_type=model_type,
            #                                 asset_data=self.all_asset_data[interval],
            #                                 c=config.fee,
            #                                 normalize_length=normalize_length,
            #                                 rnn_layers=rnn_layers,
            #                                 rnn_type=rnn_type,
            #                                 linear_base=linear_base,
            #                                 batch_length=config.batch_length,
            #                                 train_length=train_length,
            #                                 max_epoch=config.max_training_epoch,
            #                                 learning_rate=config.learning_rate,
            #                                 model_path=model_type,
            #                                 drop=drop,
            #                                 patient=config.patient,
            #                                 patient_rounds=config.patient_rounds,
            #                                 data_interval=interval,
            #                                 lr_type=lr_type,
            #                                 opt_type=opt_type)

            p = mp.Process(target=ModelTrainer.create_new_model, args=(model_type,
                                                                    self.all_asset_data[interval],
                                                                    config.fee,
                                                                    normalize_length,
                                                                    rnn_layers,
                                                                    rnn_type,
                                                                    linear_base,
                                                                    config.batch_length,
                                                                    train_length,
                                                                    config.max_training_epoch,
                                                                    config.learning_rate,
                                                                    model_type,
                                                                    drop,
                                                                    config.patient,
                                                                    config.patient_rounds,
                                                                    interval,
                                                                    lr_type,
                                                                    opt_type))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    
    def back_test(self):
        if len(self.portfolio) == 0:
            print("Init data first")
            return
        if self.asset_data is None:
            print("Load model first")
            return

        test_lengh = self.asset_data.shape[1] - config.train_length
        test_reward, test_actions =self.agent.back_test(asset_data=self.asset_data, c=config.fee,
                            test_length=test_lengh)
        market = self.asset_data.ix[:, -test_lengh:, 'diff'].cumsum()
        result = pd.DataFrame(test_reward, columns=['return'], index=market.index).cumsum()
        result['market'] = market
        return result
    
    def trade(self):
        print('=' * 100)
        if len(self.portfolio) == 0 or self.asset_data is None or self.agent is None:
            print('Init data and model')
            return
        actions = self.agent.trade(asset_data=self.asset_data)
        print('predict action for portfolio', list(zip(self.portfolio, actions)))
        self.trader.re_balance(actions=actions)
        print(datetime.datetime.now())
    
    def optimize_portfolio(self, method='CAPM', risky_number=config.risk_asset_number, risk_free_number=config.risk_free_asset_number):
        symbols = lmap(lambda x: x['base-currency'], lfilter(lambda x: x['symbol-partition'] == 'innovation' and x['quote-currency'] == BASE_CURRENCY, get_symbols()['data']))
        print('fetching data')
        data = klines(symbols, interval=config.portfolio_selection_tick_interval, count=config.portfolio_selection_bar_count)
        print('building data')
        data = OrderedDict(data)
        data = pd.Panel(data)
        data = data.dropna(axis=1)
        data.to_pickle('all_assets')
        market_index = data[:, :, 'diff'].mean(axis=1)
        if method == 'CAPM':
            print('applying CAPM')
            capm = pd.DataFrame(lmap(lambda x: linreg(x=market_index.values, y=data[x, :, 'diff'].values), data.items), index=data.items, columns=['alpha', 'beta'])
            high_risk = capm[(capm['alpha'] > 0) & (capm['beta'] > 1)].sort_values('alpha')
            low_risk = capm[(capm['alpha'] > 0) & (capm['beta'] < 1)].sort_values('alpha')
            print(len(high_risk), 'risky candidates')
            print(len(low_risk), 'risk-free candidates')
            candidate = []
            if risky_number > 0:
                candidate.extend(list(high_risk[-risky_number:].index))
            if risk_free_number > 0:
                candidate.extend(list(low_risk[-risk_free_number:].index))
            print(len(candidate))
            print(candidate)
            return candidate
        else:
            # not implemented
            return []


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please input command')
        sys.exit(1)
    command = sys.argv[1]
    portfolio_manager = PortfolioManager()
    portfolio_manager.init_assets(assets_config=config.portfolio_config)
    if command == 'trade':
        last_trade_time = None
        portfolio_manager.init_trader()
        portfolio_manager.init_data(config.trade_bar_count)
        portfolio_manager.load_model()
        print("Waiting to trade when triggered")
        while True:
            current_time = str(datetime.datetime.now().hour) + '_' + str(datetime.datetime.now().minute)
            if (datetime.datetime.now().minute in config.trade_time) and (last_trade_time != current_time):
                print("Start to trade on {0}".format(datetime.datetime.now()))
                last_trade_time = current_time
                try:
                    portfolio_manager.init_data(config.trade_bar_count)
                except Exception as e:
                    portfolio_manager.init_data(config.trade_bar_count)
                portfolio_manager.trade()
    elif command == 'trade_now':
        try:
            portfolio_manager.init_data(config.trade_bar_count)
        except Exception:
            portfolio_manager.init_data(config.trade_bar_count)
        portfolio_manager.init_trader()
        portfolio_manager.load_model()
        portfolio_manager.trade()
    
    elif command == 'build_model':
        portfolio_manager.init_data(config.train_bar_count, config.data_mode)
        portfolio_manager.build_model()
    elif command == 'build_model_batch':
        portfolio_manager.build_model_batch()
    elif command == 'backtest':
        portfolio_manager.init_data(config.train_bar_count, config.data_mode)
        portfolio_manager.load_model('model_backup/best/DRL_Torch_1gru_128_best_0.98')
        portfolio_manager.back_test()
    elif command == 'security_selection':
        raise NotImplementedError()
    else:
        print('invalid command')
        # Donate XMR:   46s5pd7bDir9dQYYH1N1g13Kb6HX9wyfDYJxFeSXHU619VBPpWH91xiPHWSimDXSanXhRZ6AyZvBv9TV69s3818DTftScBB
        # :)

    print("--- %s seconds ---" % (time.time() - start_time))
