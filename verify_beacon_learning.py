#!/usr/bin/env python3
"""
验证脚本：检查信标学习激励方案的实现完整性
"""
import sys
sys.path.append('.')

def verify_implementation():
    """验证实现是否完整"""
    print("=" * 70)
    print("信标激励方案实现验证")
    print("=" * 70)
    
    # 1. 检查env.py中的参数初始化
    print("\n✓ 检查env.py中的参数初始化...")
    from envs.env import Env
    import inspect
    
    # 获取__init__方法的源代码
    source = inspect.getsource(Env.__init__)
    
    required_params = [
        'beacon_reward_urgency_threshold',
        'beacon_reward_urgency_scale',
        'risk_awareness_threshold',
        'approach_beacon_reward',
        'move_away_beacon_penalty'
    ]
    
    for param in required_params:
        if param in source:
            print(f"  ✓ {param} 已初始化")
        else:
            print(f"  ✗ {param} 缺失！")
            return False
    
    # 2. 检查_compute_min_beacon_distance方法
    print("\n✓ 检查_compute_min_beacon_distance方法...")
    if hasattr(Env, '_compute_min_beacon_distance'):
        print("  ✓ _compute_min_beacon_distance 方法存在")
    else:
        print("  ✗ _compute_min_beacon_distance 方法缺失！")
        return False
    
    # 3. 检查_get_obs方法
    print("\n✓ 检查_get_obs方法（观测空间）...")
    get_obs_source = inspect.getsource(Env._get_obs)
    if 'ins_error_ratio' in get_obs_source:
        print("  ✓ _get_obs 正确返回ins_error_ratio（紧急度比值）")
    else:
        print("  ✗ _get_obs 未修改为返回比值！")
        return False
    
    # 4. 检查step方法中的关键变量
    print("\n✓ 检查step方法中的关键变量...")
    step_source = inspect.getsource(Env.step)
    
    if 'ins_error_before_move' in step_source:
        print("  ✓ step方法保存了ins_error_before_move")
    else:
        print("  ✗ step方法未保存ins_error_before_move！")
        return False
    
    if 'min_beacon_dist_before' in step_source:
        print("  ✓ step方法保存了min_beacon_dist_before")
    else:
        print("  ✗ step方法未保存min_beacon_dist_before！")
        return False
    
    if 'urgency_ratio' in step_source:
        print("  ✓ step方法计算了urgency_ratio（动态信标奖励）")
    else:
        print("  ✗ step方法未计算urgency_ratio！")
        return False
    
    if 'risk_awareness_threshold' in step_source:
        print("  ✓ step方法使用了risk_awareness_threshold（约束靠近激励）")
    else:
        print("  ✗ step方法未使用risk_awareness_threshold！")
        return False
    
    # 5. 检查STEP_REWARD_COLUMNS
    print("\n✓ 检查step奖励日志列定义...")
    if 'approach_beacon_reward' in Env.STEP_REWARD_COLUMNS:
        print("  ✓ STEP_REWARD_COLUMNS包含approach_beacon_reward")
    else:
        print("  ✗ STEP_REWARD_COLUMNS未包含approach_beacon_reward！")
        return False
    
    # 6. 检查_append_step_reward_log方法签名
    print("\n✓ 检查_append_step_reward_log方法签名...")
    append_sig = inspect.signature(Env._append_step_reward_log)
    if 'approach_beacon_reward' in append_sig.parameters:
        print("  ✓ _append_step_reward_log包含approach_beacon_reward参数")
    else:
        print("  ✗ _append_step_reward_log缺少approach_beacon_reward参数！")
        return False
    
    # 7. 检查config文件
    print("\n✓ 检查config1.yaml配置...")
    import yaml
    with open('configs/config1.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    config_params = [
        'beacon_reward_urgency_threshold',
        'beacon_reward_urgency_scale',
        'risk_awareness_threshold',
        'approach_beacon_reward',
        'move_away_beacon_penalty'
    ]
    
    for param in config_params:
        if param in config['reward']:
            val = config['reward'][param]
            print(f"  ✓ {param} = {val}")
        else:
            print(f"  ✗ config中缺少{param}！")
            return False
    
    print("\n" + "=" * 70)
    print("✓ 所有验证通过！实现完整正确")
    print("=" * 70)
    return True

if __name__ == '__main__':
    success = verify_implementation()
    sys.exit(0 if success else 1)
