from myestimate.hyper_graph_network import Trainer

def run_model(config, flag='train'):
    # init estimate instance
    model = Trainer(config)

    if flag == 'train':
        model.train()

    test_performance = model.test()

    return True


if __name__ == '__main__':
    config = {
        'data_dict' : {
            "group1" : ['3701', '3703', '3704', '3706', '3707', '3708'],
            "group2" : ['3501', '3502', '3503', '3504', '3505', '3506', '3507', '3508'],
            "group3" : ['1201', '1202', '1203', '1204', '1205', '1206', '1207', '1208', '1209', '1210'],
            "group4" : ['1301', '1302', '1303', '1304', '1305', '1306', '1307', '1308', '1309', '1310'],
            "group5" : ['3204', '3205', '3206', '3207', '3208', '3209'],
            "group6" : ['3304', '3307', '3308', '3309', '3310', '3311', '3312', '3313']
        },
        'cols': ['tps', 'txn_elapse', 'sql_fetch_count', 'sql_exec_count', 'sql_prepare_count', 'request_rate','jvm_cpu_usage',
                 'sql_elapse', 'active_db_sessions', 'os_cpu'],
        'remove': ['jvm_heap_usage', 'jvm_thread_count', 'open_socket', 'os_used_memory',
                   'active_txns', 'jvm_gc_time', 'jvm_gc_count']
    }
    print(f"usage feature : {len(config['cols'])}")
    print(f"un-used feature : {len(config['remove'])}")
    performances = run_model(config, flag='train')
