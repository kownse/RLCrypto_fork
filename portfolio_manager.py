# -*- coding:utf-8 -*-
import sys
import importlib
from utils.DataUtils import default_pre_process
from utils.TradingUtils import *
from utils.config import *

CONFIG_PATH = './config/config.json'
if not os.path.exists(CONFIG_PATH):
    print("config file doesn't exist")
    sys.exit(1)
init_config(CONFIG_PATH)

init_account(ACCOUNT_FILE)
print(get_accounts())


class Trader(object):
    def __init__(self):
        self.portfolio = []
        self.asset_data = None
        self.model = None
    
    def init_portfolio(self, portfolio_config):
        if not os.path.exists(portfolio_config):
            print('Portfolio config file doesn\'t exist, run PortfolioManager first')
            return
        with open(portfolio_config, 'r') as f:
            print('Load portfolio successfully')
            self.portfolio = json.loads(f.read())
    
    def init_data(self, bar_count):
        if len(self.portfolio) == 0:
            print('Load portfolio first')
            return
        asset_data = klines(self.portfolio, base_currency=BASE_CURRENCY, interval=TICK_INTERVAL, count=bar_count)
        asset_data = default_pre_process(asset_data)
        self.asset_data = asset_data
    
    def load_model(self):
        if len(self.portfolio) == 0 or self.asset_data is None:
            print('Init data first')
            return
        self.model = TRADER_MODEL(s_dim=self.asset_data.shape[-1],
                                  b_dim=self.asset_data.shape[0],
                                  a_dim=2,
                                  learning_rate=LEARNING_RATE,
                                  batch_length=BATCH_LENGTH,
                                  normalize_length=NORMALIZE_LENGTH)
        self.model.load_model(model_path=MODEL_PATH)
    
    def build_model(self):
        if len(self.portfolio) == 0 or self.asset_data is None:
            print('Init data first')
            return
        self.model = TRADER_MODEL.create_new_model(asset_data=self.asset_data,
                                                   c=FEE,
                                                   normalize_length=NORMALIZE_LENGTH,
                                                   batch_length=BATCH_LENGTH,
                                                   train_length=TRAIN_LENGTH,
                                                   max_epoch=MAX_TRAINING_EPOCH,
                                                   learning_rate=LEARNING_RATE,
                                                   pass_threshold=REWARD_THRESHOLD,
                                                   model_path=MODEL_PATH)
    
    def back_test(self):
        if len(self.portfolio) == 0 or self.asset_data is None:
            print("Init data first")
            return
        self.model.back_test(asset_data=self.asset_data, c=FEE, test_length=TEST_LENGTH)
    
    def trade(self):
        print('=' * 100)
        if len(self.portfolio) == 0 or self.asset_data is None or self.model is None:
            print('Init data and model')
            return
        actions = self.model.trade(asset_data=self.asset_data)
        print('predict action for portfolio', list(zip(self.portfolio, actions)))
        total = np.sum(actions)
        if total > 0:
            actions = np.clip(actions / total, 0, MAX_ASSET_PERCENT)
        actions = sorted(zip(self.portfolio, actions), key=lambda x: x[1])
        for asset, action in actions:
            re_balance(action,
                       symbol=asset + BASE_CURRENCY,
                       asset=asset,
                       portfolio=self.portfolio,
                       base_currency=BASE_CURRENCY,
                       order_type=ORDER_TYPE,
                       price_discount=PRICE_DISCOUNT,
                       amount_discount=AMOUNT_DISCOUNT,
                       debug=DEBUG_MODE,
                       wait_interval=ORDER_WAIT_INTERVAL,
                       trace_order=TRACE_ORDER,
                       max_order_waiting_time=MAX_ORDER_WAITING_TIME)
        print(datetime.datetime.now())


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please input command')
        sys.exit(1)
    command = sys.argv[1]
    trader = Trader()
    trader.init_portfolio(portfolio_config=PORTFOLIO_CONFIG)
    if command == 'trade':
        last_trade_hour = None
        trader.init_data(TRADE_BAR_COUNT)
        trader.load_model()
        print("Waiting to trade when triggered")
        while True:
            if datetime.datetime.now().minute == TRADE_TRIGGER and last_trade_hour != datetime.datetime.now().hour:
                print("Start to trade on {0}".format(datetime.datetime.now()))
                last_trade_hour = datetime.datetime.now().hour
                try:
                    trader.init_data(TRADE_BAR_COUNT)
                except Exception as e:
                    trader.init_data(TRADE_BAR_COUNT)
                trader.trade()
    elif command == 'trade_now':
        try:
            trader.init_data(TRADE_BAR_COUNT)
        except Exception:
            trader.init_data(TRADE_BAR_COUNT)
        trader.load_model()
        trader.trade()
        
    elif command == 'build_model':
        trader.init_data(TRAIN_BAR_COUNT)
        trader.build_model()
    elif command == 'backtest':
        trader.init_data(TRAIN_BAR_COUNT)
        trader.load_model()
        trader.back_test()
    else:
        print('invalid command')
        # Donate XMR:   4AUY1FEpfGtYutRShAsmTMbVFmLoZdL92Gg6fQPYsN1P61mqrZpgnmsQKtYM8CkFpvDMJS6MuuKmncHhSpUtRyEqGcNUht2
        # :)