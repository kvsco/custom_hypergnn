import torch
import datetime

from myestimate.hyper_graph_network import Trainer

def run_model(config, flag='train'):
    # init estimate instance
    model = Trainer(config)

    if flag == 'train':
        setting = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        model.train(setting)
        model.test(setting)
    torch.cuda.empty_cache()
    return True


if __name__ == '__main__':
    config = {
        'lookback_window' : 60,
        'lookahead_window' : 30,
        'data_dict' : {
            "group1" : ['3701', '3703', '3704', '3706', '3707', '3708'],
            "group2" : ['3501', '3502', '3503', '3504', '3505', '3506', '3507', '3508'],
            "group3" : ['1201', '1202', '1203', '1204', '1205', '1206', '1207', '1208', '1209', '1210'],
            "group4" : ['1301', '1302', '1303', '1304', '1305', '1306', '1307', '1308', '1309', '1310'],
            "group5" : ['3204', '3205', '3206', '3207', '3208', '3209'],
            "group6" : ['3304', '3307', '3308', '3309', '3310', '3311', '3312', '3313']
        },
        'cols' : ['txn_elapse', 'jvm_cpu_usage', 'active_txns', 'request_rate', 'active_db_sessions', 'sql_exec_count',
                'sql_elapse', 'sql_prepare_count', 'sql_fetch_count', 'os_cpu', 'tps', 'jvm_gc_time', 'jvm_gc_count'],
        'columns_to_exclude' : ['open_socket_count', 'os_used_memory', 'jvm_heap_usage', 'jvm_thread_count']
    }
    print(f"usage feature : {len(config['cols'])}")
    print(f"un-used feature : {len(config['columns_to_exclude'])}")
    performances = run_model(config, flag='train')
