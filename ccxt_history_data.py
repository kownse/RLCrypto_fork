import os
import os.path
from os import path
import sys
import time
import datetime
import pandas as pd
from tqdm import tqdm
import ccxt  # noqa: E402

def download_history(exchange, symble, from_datetime, end_datetime, interval='1h'):
    data_path = "data/{0}".format(interval)
    if not path.exists(data_path):
        try:
            os.makedirs(data_path)
        except OSError:
            print ("Creation of the directory %s failed" % data_path)
        else:
            print ("Successfully created the directory %s" % data_path)

    from_timestamp = exchange.parse8601(from_datetime)
    initial_timestamp = from_timestamp
    end = exchange.parse8601(end_datetime)
    data = []
    
    if interval=='1h':
        delta = minute * 60
    else:
        print("unsupported interval:" + interval)
        exit()
            
    tot = (int)((end - from_timestamp) / delta)
    retry = 0
    pbar = None

    while from_timestamp < end:
        try:
            ohlcvs = exchange.fetch_ohlcv(symble, interval, from_timestamp)
            if len(ohlcvs) > 0:
                if pbar is None:
                    first = ohlcvs[0][0]
                    tot = (int)((end - first) / delta)
                    pbar = tqdm(total=tot)
                    print('start downloading', symble, 'from', datetime.datetime.utcfromtimestamp(first/1000), 'total', tot)
                from_timestamp = ohlcvs[-1][0] + delta  # good
                data += ohlcvs
                if pbar is not None:
                    pbar.update(len(ohlcvs))
            else:
                # print('can not fetch more data, finish...')
                break
        except (ccxt.ExchangeError, ccxt.AuthenticationError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as error:
            retry = retry + 1
            if retry > max_retry:
                print('Retried {0} times, give up...'.format(retry))
                break
            #print('Got an error', type(error).__name__, error.args, ', retrying in', hold, 'seconds...')
            time.sleep(hold)
    if pbar is not None:
        pbar.close()
    if len(data) > 0:
        header = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(data, columns=header)
        save_path = '{0}/{1}_{2}_{3}.csv'.format(data_path, symble.replace('/','-'), initial_timestamp, from_timestamp)
        df.to_csv(save_path, index=False)

if __name__=='__main__':

    msec = 1000
    minute = 60 * msec
    hold = 15
    max_retry = 5

    exchange = ccxt.bitfinex({
        'rateLimit': 10000,
        'enableRateLimit': True,
    })
    exchange.load_markets()
    usd_symbles = [symble for symble in exchange.symbols if symble[-3:] == 'USD']
    for symble in tqdm(usd_symbles):
        download_history(exchange, symble, '2005-05-29 00:00:00', '2019-12-22 00:00:00', '1h')
