import torch

def compute_groups(world_size, tp, pp, dp):
    assert world_size % (tp * pp * dp) == 0
    E = world_size // (tp * pp * dp)  # ExternalDP 大小（通常为 1）
    all_ranks = torch.arange(world_size).reshape(E, dp, pp, tp)

    # TP 组
    tp_groups = [x.tolist() for x in all_ranks.view(-1, tp).unbind(0)]

    # PP 组
    pp_groups = [x.tolist() for x in all_ranks.transpose(2, 3).reshape(-1, pp).unbind(0)]

    # DP 组（模型内部 DP）
    dp_groups = [x.tolist() for x in all_ranks.transpose(1, 3).reshape(-1, dp).unbind(0)]

    # EP 组（同 PP stage 下合并 DP×TP）
    ep_groups = [x.tolist() for x in all_ranks.transpose(1, 2).reshape(-1, dp * tp).unbind(0)]

    print("TP groups:", tp_groups)
    print("PP groups:", pp_groups)
    print("DP groups:", dp_groups)
    print("EP groups:", ep_groups)

if __name__ == "__main__":
    """
    world_size = 16
    tp_size = 4
    pp_size = 1
    dp_size = 4
    """
    compute_groups(4, 2, 2, 1) # 卡数 4