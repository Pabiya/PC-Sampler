import torch
import numpy as np
import torch.nn.functional as F
import os, sys

current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(scripts_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

BASE_LINE= None

def entropy_function(probabilities):
    if probabilities.dim() != 3:
        raise ValueError("Input tensor 'probabilities' must be a 3D tensor with shape [batch_size, sequence_len, vocab_size]")
    epsilon = 1e-12
    probs_safe = probabilities.clone() + epsilon
    entropy = torch.sum(probabilities.clone() * torch.log(probs_safe), dim=-1)
    return entropy

def margin_function(probabilities):
    if probabilities.dim() != 3:
        raise ValueError("Input tensor 'probabilities' must be a 3D tensor with shape [batch_size, sequence_len, vocab_size]")
    sorted_probs, _ = torch.sort(probabilities, dim=-1, descending=True)
    top1_probs = sorted_probs[:, :, 0]
    top2_probs = sorted_probs[:, :, 1]
    confidence = top1_probs - top2_probs
    return confidence

def pc_sampler_function(
    probabilities: torch.Tensor,
    token_ids: torch.Tensor,
    lambda_val: float,
    alpha: float,
    bg_freq_tensor: torch.Tensor
) -> torch.Tensor:
    
    if probabilities.shape != token_ids.shape:
        raise f"probabilities.shape: {probabilities.shape}, token_ids.shape: {token_ids.shape} must be equal"

    device = probabilities.device
    sequence_len = probabilities.shape[1]
    f_bg_tensor = bg_freq_tensor[token_ids]
    epsilon = 1e-9
    cross_entropy_scores = -probabilities * torch.log(f_bg_tensor + epsilon)
    cross_entropy_scores = torch.clamp(cross_entropy_scores, max=alpha)
    positions = torch.arange(sequence_len, device=device, dtype=torch.float32)
    positional_bias = torch.exp(-lambda_val * positions)
    final_scores = positional_bias * cross_entropy_scores

    return final_scores

def linear_position(
    probabilities: torch.Tensor,
    token_ids: torch.Tensor,
    lambda_val: float,
    alpha: float,
    bg_freq_tensor: torch.Tensor
) -> torch.Tensor:

    if probabilities.shape != token_ids.shape:
        raise f"probabilities.shape: {probabilities.shape}, token_ids.shape: {token_ids.shape} must be equal"
    device = probabilities.device
    batch_size, sequence_len = probabilities.shape
    f_bg_tensor = bg_freq_tensor[token_ids]
    epsilon = 1e-9
    cross_entropy_scores = -probabilities * torch.log(f_bg_tensor + epsilon)
    confidence = torch.clamp(cross_entropy_scores, max=alpha)
    conf_max = confidence.view(batch_size, -1).max(dim=1, keepdim=True)[0].unsqueeze(1)
    conf_min = confidence.view(batch_size, -1).min(dim=1, keepdim=True)[0].unsqueeze(1)
    denom = conf_max - conf_min
    denom[denom < 1e-9] = 1e-9
    confidence_normalized = (confidence - conf_min) / denom
    positions = torch.arange(sequence_len, device=device, dtype=torch.float32)
    positional_bias = (sequence_len - positions) / sequence_len
    final_scores = (1 - lambda_val) * confidence_normalized + lambda_val * positional_bias
    return final_scores

def load_baseline(model, baseline_name):
    global BASE_LINE
    if BASE_LINE is None:
        from utils.load_json_or_jsonl import load_json_or_jsonl
        p_baseline_dict = load_json_or_jsonl(baseline_name)
        token_num_ = p_baseline_dict['num_token']
        p_baseline_dict = p_baseline_dict['p_baseline_dict']
        del_keys = []
        for key in p_baseline_dict.keys():
            del_keys.append(key)
        for key in del_keys:
            p_baseline_dict[int(key)] = p_baseline_dict[key]
        for key in del_keys:
            del p_baseline_dict[key]
        for key in p_baseline_dict.keys():
            p_baseline_dict[key] = p_baseline_dict[key] / token_num_
        BASE_LINE = torch.full((126464,), 1/token_num_, device=model.device, dtype=torch.float32)
        keys = torch.tensor(list(p_baseline_dict.keys()), device=model.device, dtype=torch.long)
        values = torch.tensor(list(p_baseline_dict.values()), device=model.device, dtype=torch.float32)
        BASE_LINE.scatter_(0, keys, values)
    else:
        BASE_LINE = BASE_LINE.to(model.device)

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index

@torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, return_order=False):

    
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)
    if return_order:
        orders = {}
    
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    
    assert steps % num_blocks == 0
    steps = steps // num_blocks
    
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length:prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
                if return_order:
                    if num_block+1 not in orders:
                        orders[num_block+1] = []
                    orders[num_block+1].append((select_index-prompt.shape[1]).tolist())
            x[transfer_index] = x0[transfer_index]
    if return_order:
        return x, orders        
    return x

# @torch.no_grad()
# def generate_with_refine_ent3(
#     model,
#     prompt,
#     steps=128,
#     gen_length=128,
#     block_length=128,
#     temperature=0.0,
#     remasking='low_confidence',
#     mask_id=126336,
#     refine_every=1,
#     nucleus_p=1.0,
#     return_order=False,
# ):
#     """
#     LLaDA용 refine-ent-3 (B-type) 샘플러:
#       1) 각 스텝에서 기본 전이로 일부 토큰 언마스크
#       2) 주기(refine_every)마다 R 위치를 선택해 2nd-forward로 즉시 refine
#       3) refine 직후 분포로 해당 위치의 entropy를 재계산하여 confidence 캐시에 '덮어쓰기'
#     """
#     device = model.device
#     B = 1
#     x = torch.full((B, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
#     x[:, :prompt.shape[1]] = prompt.clone()

#     if return_order:
#         orders = {}

#     assert gen_length % block_length == 0
#     num_blocks = gen_length // block_length

#     assert steps % num_blocks == 0
#     steps_per_block = steps // num_blocks

#     # dtype을 float32로 통일해 확률/엔트로피 계산의 안정성 확보
#     fp = torch.float32
#     eps = torch.finfo(fp).eps

#     # confidence 캐시 (entropy). 초기에는 매우 낮게 설정
#     conf = torch.full_like(x, fill_value=-1e9, dtype=fp)

#     step_counter = 0

#     for num_block in range(num_blocks):
#         blk_lo = prompt.shape[1] + num_block * block_length
#         blk_hi = prompt.shape[1] + (num_block + 1) * block_length
#         block_mask_index = (x[:, blk_lo:blk_hi] == mask_id)
#         num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

#         for i in range(steps_per_block):
#             mask_index = (x == mask_id)

#             # predictor
#             logits = model(x).logits.to(fp)  # float32 변환
#             logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
#             x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L)

#             # confidence(=1-entropy 아님! 아래에서 entropy 계산 전까진 top1 확률로 사용)
#             if remasking == 'low_confidence':
#                 p = F.softmax(logits, dim=-1)  # fp32
#                 x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1).to(fp)  # (B, L)
#             elif remasking == 'random':
#                 x0_p = torch.rand_like(x0, dtype=fp)
#             else:
#                 raise NotImplementedError(remasking)

#             # 블록 밖은 아직 잠금
#             x0_p[:, blk_hi:] = -np.inf
#             x0 = torch.where(mask_index, x0, x)
#             confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=device, dtype=fp))

#             # 해당 스텝에서 확정할 개수만 선택
#             transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
#             for b in range(B):
#                 k = int(num_transfer_tokens[b, i].item())
#                 if k > 0:
#                     _, sel = torch.topk(confidence[b], k=k)
#                     transfer_index[b, sel] = True
#                     if return_order:
#                         blk = num_block + 1
#                         if blk not in orders:
#                             orders[blk] = []
#                         orders[blk].append((sel - prompt.shape[1]).tolist())
#             x[transfer_index] = x0[transfer_index]

#             # --- (B) in-step 2-forward refine (refine_every 간격) ---
#             if (step_counter % max(1, refine_every)) == 0:
#                 # 1) 확정된 위치들 중 [MASK]가 아닌 곳만 대상으로 entropy 기반 선택 가중치
#                 with torch.no_grad():
#                     p1 = F.softmax(logits, dim=-1).clamp_min(eps)  # fp32
#                     p1_wo = p1.clone()
#                     p1_wo[..., mask_id] = 0.0
#                     Z = p1_wo.sum(dim=-1, keepdim=True).clamp_min(eps)
#                     q = p1_wo / Z
#                     H = -(q * (q + eps).log()).sum(dim=-1)  # (B, L) entropy, fp32

#                 # [MASK]였던 곳 제외
#                 unmasked_flag = (x != mask_id)
#                 eta = torch.softmax(H, dim=-1).masked_fill(~unmasked_flag, 0.0)  # 확정 위치만 후보
#                 # sigma = eta * sigma_max → LLaDA에는 명시적 schedule이 없으므로 sigma_max=1.0 사용
#                 sigma = eta.clamp(0.0, 1.0)
#                 R = (torch.rand_like(sigma) < sigma) & unmasked_flag
#                 if R.any():
#                     # 2) R만 remask하여 2nd forward
#                     x_tmp = x.clone()
#                     x_tmp[R] = mask_id
#                     logits2 = model(x_tmp).logits.to(fp)  # fp32

#                     # nucleus(top-p) 옵션
#                     if nucleus_p < 1.0:
#                         p2 = F.softmax(logits2, dim=-1)  # fp32
#                         sorted_probs, sorted_idx = torch.sort(p2, dim=-1, descending=True)
#                         cprob = torch.cumsum(sorted_probs, dim=-1)
#                         keep = (cprob <= nucleus_p)
#                         keep[..., 0] = True
#                         nucleus = sorted_probs * keep
#                         nucleus = nucleus / nucleus.sum(dim=-1, keepdim=True).clamp_min(eps)
#                         p_x0_2 = torch.zeros_like(p2).scatter_(-1, sorted_idx, nucleus)
#                     else:
#                         p_x0_2 = F.softmax(logits2, dim=-1)  # fp32

#                     # 3) R 위치만 재샘플 → 즉시 덮어쓰기
#                     sampled2 = torch.argmax(
#                         add_gumbel_noise(torch.log(p_x0_2.clamp_min(eps)), temperature=0.0),
#                         dim=-1
#                     )
#                     x[R] = sampled2[R]

#                     # 4) (핵심) refine 직후 분포의 entropy로 conf 캐시 덮어쓰기
#                     P2 = p_x0_2.clamp_min(eps)  # fp32
#                     P2[..., mask_id] = 0.0
#                     Z2 = P2.sum(dim=-1, keepdim=True).clamp_min(eps)
#                     Q2 = P2 / Z2
#                     H2 = -(Q2 * (Q2 + eps).log()).sum(dim=-1)
#                     conf[R] = H2[R].to(conf.dtype)

#             step_counter += 1

#     if return_order:
#         return x, orders
#     return x

@torch.no_grad()
def generate_with_refine_ent3(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking='low_confidence',   # 기본 전이는 기존 LLaDA의 선택 기준 그대로 사용
    mask_id=126336,
    refine_every=1,               # k 스텝마다 1회 실행 (k=1이면 매 스텝)
    nucleus_p=1.0,                # 2nd forward에도 동일 top-p 적용
    refine_select="multinomial",  # "multinomial" | "topk"
    refine_K=None,                # None -> 이번 스텝의 nun(언마스크 예산) 사용, 아니면 고정 K(스텝마다 min(K, nun))
    entropy_remove_mask_prob=True,# 엔트로피 계산 시 [MASK] 확률 제거/재정규화
    cfg_scale=0.0,                # CFG 스케일(0이면 비활성)
    return_order=False,
):
    """
    LLaDA + MIRAGE (옵션 A, C=프롬프트 제외 전체 생성 구간의 언마스크 위치)
      1) LLaDA 전이로 이번 스텝 언마스크 실행
      2) (주기적으로) C에서 R을 선택 → R만 remask → 2nd forward → R만 재언마스크
      3) conf 캐시: [MASK]->token 된 곳은 H1, refine된 R은 H2로 덮어쓰기
    """
    device = model.device
    B = 1
    P = prompt.shape[1]
    L = gen_length
    fp = torch.float32
    eps = torch.finfo(fp).eps

    # 고정 프롬프트 마스크 (프롬프트 위치 True)
    x = torch.full((B, P + L), mask_id, dtype=torch.long, device=device)
    x[:, :P] = prompt.clone()
    prompt_mask_fixed = torch.zeros_like(x, dtype=torch.bool)
    prompt_mask_fixed[:, :P] = True

    if return_order:
        orders = {}

    assert L % block_length == 0
    num_blocks = L // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    # ── CFG/로짓 헬퍼 (fp32 반환)
    def forward_logits(seq):
        if cfg_scale and cfg_scale > 0.0:
            un_x = seq.clone()
            un_x[~prompt_mask_fixed] = un_x[~prompt_mask_fixed]  # no-op, 가독성용
            un_x[prompt_mask_fixed] = mask_id                    # 프롬프트 제거한 언컨디셔닝
            x_cat = torch.cat([seq, un_x], dim=0)                # (2, T)
            logits = model(x_cat).logits.to(fp)
            cond, uncond = torch.chunk(logits, 2, dim=0)         # (1,T,V) each
            return (uncond + (cfg_scale + 1.0) * (cond - uncond)).to(fp)
        else:
            return model(seq).logits.to(fp)

    # 엔트로피 도우미
    def entropy_from_probs(P, remove_mask_prob=True):
        # P: (B, T, V) 확률, fp32
        P = P.clamp_min(eps)
        if remove_mask_prob:
            P_wo = P.clone()
            P_wo[..., mask_id] = 0.0
            Z = P_wo.sum(dim=-1, keepdim=True).clamp_min(eps)
            Q = P_wo / Z
        else:
            Z = P.sum(dim=-1, keepdim=True).clamp_min(eps)
            Q = P / Z
        H = -(Q * (Q + eps).log()).sum(dim=-1)  # (B, T)
        return H

    # conf 캐시(엔트로피): 처음엔 -inf
    conf = torch.full((B, P + L), -float("inf"), dtype=fp, device=device)

    step_counter = 0

    for num_block in range(num_blocks):
        blk_lo = P + num_block * block_length
        blk_hi = P + (num_block + 1) * block_length

        block_mask_index = (x[:, blk_lo:blk_hi] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)  # (B, steps_per_block)

        for i in range(steps_per_block):
            # ────────────── (1) 기본 LLaDA 전이: 이번 스텝 언마스크 ──────────────
            mask_index = (x == mask_id)

            logits = forward_logits(x)                     # (B, T, V)
            if temperature > 0.0:
                logits = logits / max(1e-6, temperature)

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)   # (B, T)

            if remasking == 'low_confidence':
                p1 = F.softmax(logits, dim=-1)             # (B, T, V) fp32
                x0_p = torch.gather(p1, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (B, T)
            elif remasking == 'random':
                p1 = F.softmax(logits, dim=-1)             # 그래도 p1은 뒤에서 엔트로피용으로 사용
                x0_p = torch.rand_like(x0, dtype=fp)
            else:
                raise NotImplementedError(remasking)

            # 블록 경계 밖은 잠금
            x0_p[:, blk_hi:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=device, dtype=fp))

            # 이번 스텝 nun만큼 선택
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
            for b in range(B):
                nun = int(num_transfer_tokens[b, i].item())
                if nun > 0:
                    _, sel = torch.topk(confidence[b], k=nun)
                    transfer_index[b, sel] = True
                    if return_order:
                        blk = num_block + 1
                        if blk not in orders:
                            orders[blk] = []
                        orders[blk].append((sel - P).tolist())

            # 언마스크 적용
            became_unmasked = transfer_index & (x == mask_id)
            x[transfer_index] = x0[transfer_index]

            # --- H1 워밍업: 이번 스텝 막 언마스크된 위치의 엔트로피 기록 ---
            H1 = entropy_from_probs(p1, remove_mask_prob=entropy_remove_mask_prob)
            conf[became_unmasked] = H1[became_unmasked]

            # ────────────── (2) MIRAGE: in-step 2nd forward (주기적) ──────────────
            if (step_counter % max(1, refine_every)) == 0:
                # 후보 C: 프롬프트 제외 & 현재 언마스크된 전체 생성 구간
                C = (~prompt_mask_fixed) & (x != mask_id)  # (B, T)

                # η = softmax(conf) on C (전부 -inf면 fallback)
                eta_in = torch.where(C, conf, torch.tensor(-float('inf'), device=device, dtype=fp))
                eta = torch.softmax(eta_in, dim=-1)        # (B, T)

                bad_rows = ~torch.isfinite(eta.sum(dim=-1))
                if bad_rows.any():
                    eta_fb = torch.softmax(
                        torch.where(C, H1, torch.tensor(-float('inf'), device=device, dtype=fp)),
                        dim=-1
                    )
                    eta[bad_rows] = eta_fb[bad_rows]

                # 이번 스텝 리파인 예산 K_eff (None이면 nun 사용)
                K_vec = []
                for b in range(B):
                    nun = int(num_transfer_tokens[b, i].item())
                    k_req = nun if refine_K is None else int(refine_K)
                    # 후보 수로 클립
                    k_eff = min(k_req, int(C[b].sum().item()))
                    K_vec.append(k_eff)
                K_vec = torch.tensor(K_vec, device=device, dtype=torch.long)

                # 배치별 선택 마스크 R
                R = torch.zeros_like(C, dtype=torch.bool, device=device)
                for b in range(B):
                    k = int(K_vec[b].item())
                    if k <= 0:
                        continue
                    if refine_select == "topk":
                        _, idx = torch.topk(eta[b], k=k)
                    else:
                        # multinomial(무중복). 확률합 1이어야 함
                        idx = torch.multinomial(eta[b], num_samples=k, replacement=False)
                    R[b, idx] = True
                R = R & C  # 안전

                if R.any():
                    # R만 remask한 컨텍스트로 2nd forward
                    x_tmp = x.clone()
                    x_tmp[R] = mask_id

                    logits2 = forward_logits(x_tmp).to(fp)

                    # nucleus(top-p) 적용
                    if nucleus_p < 1.0:
                        p2 = F.softmax(logits2, dim=-1)
                        sorted_probs, sorted_idx = torch.sort(p2, dim=-1, descending=True)
                        cprob = torch.cumsum(sorted_probs, dim=-1)
                        keep = (cprob <= nucleus_p)
                        keep[..., 0] = True
                        nucleus = sorted_probs * keep
                        nucleus = nucleus / nucleus.sum(dim=-1, keepdim=True).clamp_min(eps)
                        p_x0_2 = torch.zeros_like(p2).scatter_(-1, sorted_idx, nucleus)
                    else:
                        p_x0_2 = F.softmax(logits2, dim=-1)

                    # R 위치만 재샘플 → 즉시 덮어쓰기
                    sampled2 = torch.argmax(
                        add_gumbel_noise(torch.log(p_x0_2.clamp_min(eps)), temperature=0.0),
                        dim=-1
                    )
                    x[R] = sampled2[R]

                    # H2로 conf 덮어쓰기
                    H2 = entropy_from_probs(p_x0_2, remove_mask_prob=entropy_remove_mask_prob)
                    conf[R] = H2[R]

            step_counter += 1

    if return_order:
        return x, orders
    return x

@torch.no_grad()
def generate_with_pc_sampler(model, prompt, steps=128, gen_length=128, block_length=128, lambd=1, alpha=1, baseline_name='P_baseline.json', temperature=0.,
                  cfg_scale=0., remasking='low_confidence', mask_id=126336, return_order=False):
    
    global BASE_LINE
    if BASE_LINE is None:
        load_baseline(model, baseline_name)
    if return_order:
        orders = {}
    
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            
            x0_p = pc_sampler_function(
                probabilities=x0_p[:, prompt.shape[1] + num_block * block_length:prompt.shape[1] + (num_block + 1) * block_length],
                token_ids=x0[:, prompt.shape[1] + num_block * block_length:prompt.shape[1] + (num_block + 1) * block_length],
                lambda_val=lambd,
                alpha=alpha,
                bg_freq_tensor=BASE_LINE
            )
            
            confidence = torch.where(mask_index[:, prompt.shape[1] + num_block * block_length:prompt.shape[1] + (num_block + 1) * block_length], x0_p, -np.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index+prompt.shape[1]+num_block*block_length] = True
                if return_order:
                    if num_block+1 not in orders:
                        orders[num_block+1] = []
                    orders[num_block+1].append(select_index.tolist())
            x[transfer_index] = x0[transfer_index]
    if return_order:
        return x, orders
    return x

@torch.no_grad()
def generate_with_eb_sampler(model, prompt, gamma=0.1, gen_length=128, temperature=0.,
                       cfg_scale=0., mask_id=126336):
    
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)

    while (x == mask_id).any():
        
        mask_index = (x == mask_id)
        
        if cfg_scale > 0.:
            un_x = x.clone()
            un_x[prompt_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            logits = model(x_).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(x).logits

        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        predicted_tokens = torch.argmax(logits_with_noise, dim=-1)
        masked_logits = logits[mask_index]
        
        err_proxy = torch.distributions.Categorical(logits=masked_logits).entropy()

        masked_token_indices = mask_index.nonzero(as_tuple=True)[1]
        sorted_err_indices = torch.argsort(err_proxy)
        sorted_indices = masked_token_indices[sorted_err_indices]
        
        sorted_entropies = err_proxy[sorted_err_indices]
        
        acc_entropy = torch.cumsum(sorted_entropies, dim=0)
        cummax_entropy = torch.cummax(sorted_entropies, dim=0).values
        
        k = (acc_entropy - cummax_entropy <= gamma).sum()
        
        num_masks_available = len(sorted_indices)
        k = torch.clamp(k, min=1, max=num_masks_available)

        indices_to_unmask = sorted_indices[:k]
        
        x[0, indices_to_unmask] = predicted_tokens[0, indices_to_unmask]

    return x

@ torch.no_grad()
def generate_with_fast_dllm(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
            x[transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    return x, nfe

@torch.no_grad()
def generate_with_entropy(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                            cfg_scale=0., remasking='low_confidence', mask_id=126336, return_order=False):
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    if return_order:
        orders = {}

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            x0_p = entropy_function(p[:, prompt.shape[1]:])
            confidence = torch.where(mask_index[:, prompt.shape[1]:], x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index + prompt.shape[1]] = True
                if return_order:
                    if num_block+1 not in orders:
                        orders[num_block+1] = []
                    orders[num_block+1].append(select_index.tolist())
            x[transfer_index] = x0[transfer_index]
    if return_order:
        return x, orders
    return x

@torch.no_grad()
def generate_with_margin(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                            cfg_scale=0., remasking='low_confidence', mask_id=126336, return_order=False):
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    if return_order:
        orders = {}

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            x0_p = margin_function(p[:, prompt.shape[1]:])
            confidence = torch.where(mask_index[:, prompt.shape[1]:], x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index + prompt.shape[1]] = True
                if return_order:
                    if num_block+1 not in orders:
                        orders[num_block+1] = []
                    orders[num_block+1].append(select_index.tolist())
            x[transfer_index] = x0[transfer_index]
    if return_order:
        return x, orders
    return x

@torch.no_grad()
def generate_with_linear_position(model, prompt, steps=128, gen_length=128, block_length=128, lambd=1, alpha=1, baseline_name='P_baseline.json', temperature=0.,
                  cfg_scale=0., remasking='low_confidence', mask_id=126336, return_order=False):
    
    global BASE_LINE
    if BASE_LINE is None:
        load_baseline(model, baseline_name)    
    
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    if return_order:
        orders = {}

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            
            x0_p = linear_position(
                probabilities=x0_p[:, prompt.shape[1] + num_block * block_length:prompt.shape[1] + (num_block + 1) * block_length],
                token_ids=x0[:, prompt.shape[1] + num_block * block_length:prompt.shape[1] + (num_block + 1) * block_length],
                lambda_val=lambd,
                alpha=alpha,
                bg_freq_tensor=BASE_LINE
            )
            
            confidence = torch.where(mask_index[:, prompt.shape[1] + num_block * block_length:prompt.shape[1] + (num_block + 1) * block_length], x0_p, -np.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index+prompt.shape[1]+num_block*block_length] = True
                if return_order:
                    if num_block+1 not in orders:
                        orders[num_block+1] = []
                    orders[num_block+1].append(select_index.tolist())
            x[transfer_index] = x0[transfer_index]
    if return_order:
        return x, orders
    return x

@torch.no_grad()
def generate_with_remdm(
    model,
    prompt,
    gen_length=32,
    init_unmask_ratio=0.875,   # 28/32
    unmask_k=1,                # kept for API compatibility (unused in ReMDM-conf)
    loop_steps=32,
    temperature=0.0,           # per-position sampling temp (set >0 if you want more randomness)
    cfg_scale=0.0,
    remasking='low_confidence',# kept for API compatibility (unused in ReMDM-conf)
    mask_id=126336,
    tokenizer=None,
):
    """
    dtype-safe (bf16 모델 대응) ReMDM-heuristic 버전
    - 확률/엔트로피 계산 경로를 float32로 통일
    - torch.where 상수도 dtype/device 맞춰 생성
    """
    device = model.device if hasattr(model, "device") else prompt.device
    fp = torch.float32
    neg_inf = torch.tensor(-float('inf'), device=device, dtype=fp)
    pos_inf = torch.tensor(float('inf'), device=device, dtype=fp)

    assert 0.0 <= init_unmask_ratio <= 1.0, "init_unmask_ratio must be between 0 and 1"
    num_initial_tokens = gen_length * init_unmask_ratio
    assert num_initial_tokens == int(num_initial_tokens), "gen_length * init_unmask_ratio must be an integer"
    num_initial_tokens = int(num_initial_tokens)
    assert gen_length % unmask_k == 0, "gen_length must be divisible by unmask_k"
    assert num_initial_tokens % unmask_k == 0, "init_unmask_ratio * gen_length must be divisible by unmask_k"

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)

    # ---------- 1) 초기 언마스킹 루프 ----------
    num_loops = num_initial_tokens // unmask_k
    for _ in range(num_loops):
        mask_index = (x == mask_id)
        if not mask_index.any():
            break

        # CFG (옵션) + logits → fp32
        if cfg_scale > 0.:
            un_x = x.clone()
            un_x[prompt_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            logits = model(x_).logits.to(fp)
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(x).logits.to(fp)

        if temperature > 0.0:
            logits = logits / max(1e-6, temperature)

        logits_with_noise = add_gumbel_noise(logits, temperature=0.0)  # noise는 logits에만
        x0 = torch.argmax(logits_with_noise, dim=-1)  # (1, L)

        p = F.softmax(logits, dim=-1)  # fp32
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (1, L), fp32
        confidence = torch.where(mask_index, x0_p, neg_inf)  # dtype/device 일치

        _, select_indices = torch.topk(confidence, k=unmask_k, dim=1)
        x[0, select_indices] = x0[0, select_indices]

        if tokenizer is not None:
            _ = tokenizer.decode(x[0, prompt.shape[1]:], skip_special_tokens=False)

    # ---------- 2) 리마스킹/재언마스킹 루프 ----------
    for _ in range(loop_steps):
        unmasked_gen_index = (x != mask_id) & (~prompt_index)
        num_unmasked_gen = int(torch.sum(unmasked_gen_index).item())
        if num_unmasked_gen == 0:
            continue
        current_remask_k = min(unmask_k, num_unmasked_gen)

        # 현재 토큰의 확신도 계산 (fp32)
        if cfg_scale > 0.:
            un_x = x.clone()
            un_x[prompt_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            logits = model(x_).logits.to(fp)
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(x).logits.to(fp)

        if temperature > 0.0:
            logits = logits / max(1e-6, temperature)

        p = F.softmax(logits, dim=-1)  # fp32
        # 현재 선택된 토큰의 확률
        current_token_p = torch.gather(p, dim=-1, index=x.unsqueeze(-1)).squeeze(-1)  # (1, L), fp32
        # 리마스크 후보: 확신도가 낮은 위치
        confidence = torch.where(unmasked_gen_index, current_token_p, pos_inf)
        _, remask_indices = torch.topk(confidence, k=current_remask_k, dim=1, largest=False)
        x[0, remask_indices] = mask_id

        # 리마스크 후 다시 언마스크
        mask_index_after_remask = (x == mask_id)

        if cfg_scale > 0.:
            un_x = x.clone()
            un_x[prompt_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            logits = model(x_).logits.to(fp)
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(x).logits.to(fp)

        if temperature > 0.0:
            logits = logits / max(1e-6, temperature)

        logits_with_noise = add_gumbel_noise(logits, temperature=0.0)
        x0 = torch.argmax(logits_with_noise, dim=-1)
        p_new = F.softmax(logits, dim=-1)  # fp32
        x0_p_new = torch.gather(p_new, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (1, L), fp32

        confidence_for_unmasking = torch.where(mask_index_after_remask, x0_p_new, neg_inf)
        _, unmask_indices = torch.topk(confidence_for_unmasking, k=current_remask_k, dim=1)
        x[0, unmask_indices] = x0[0, unmask_indices]

        if tokenizer is not None:
            _ = tokenizer.decode(x[0, prompt.shape[1]:], skip_special_tokens=False)

    # ---------- 3) 잔여 [MASK] 마무리 ----------
    while (x == mask_id).any():
        mask_index = (x == mask_id)
        num_masked_left = int(torch.sum(mask_index).item())
        if num_masked_left == 0:
            break
        current_unmask_k = min(unmask_k, num_masked_left)

        if cfg_scale > 0.:
            un_x = x.clone()
            un_x[prompt_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            logits = model(x_).logits.to(fp)
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(x).logits.to(fp)

        if temperature > 0.0:
            logits = logits / max(1e-6, temperature)

        logits_with_noise = add_gumbel_noise(logits, temperature=0.0)
        x0 = torch.argmax(logits_with_noise, dim=-1)
        p = F.softmax(logits, dim=-1)  # fp32
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (1, L), fp32

        confidence = torch.where(mask_index, x0_p, neg_inf)
        _, select_indices = torch.topk(confidence, k=current_unmask_k, dim=1)
        x[0, select_indices] = x0[0, select_indices]

        if tokenizer is not None:
            _ = tokenizer.decode(x[0, prompt.shape[1]:], skip_special_tokens=False)

    return x


@torch.no_grad()
def generate_with_remdm_paper(
    model,
    prompt: torch.Tensor,                 # (1, P)
    gen_length: int = 32,                 # 논문 세팅
    init_unmask_ratio: float = 0.875,     # 28/32 = 0.875
    loop_steps: int = 32,                 # 논문 세팅
    unmask_k: int = 1,                    # Countdown=1, TruthfulQA=4 등
    temperature: float = 0.0,             # 논문에선 greedy + 일부 실험에서 랜덤
    cfg_scale: float = 0.0,               # 0이면 CFG 비활성
    mask_id: int = 126336,                # LLaDA/LLaMA-류 마스크 토큰 id (환경에 맞춰 교체)
    tokenizer=None,                       # 필요시 후처리용(반환값에는 미사용)
):
    """
    Paper-style ReMDM-on-LLaDA sampler (deterministic K-remask loop):

    Phase A (Initial fill): init_unmask_ratio*gen_length 만큼 기존 LLaDA 방식으로
      [MASK] 중 상위-K 확신 위치를 선택해 채움. (K씩 반복)

    Phase B (Loop, alpha-fixed-like): loop_steps 번 반복하며,
      1) 현재 생성된(프롬프트 제외) 토큰 중 confidence 최하위 K개를 리마스크
      2) 곧바로 [MASK] 중에서 confidence 최상위 K개를 재언마스크
      (K를 보존하여 난이도/마스크 예산을 고정 = α-고정에 가까운 루프)

    Phase C (Finalize): 남은 [MASK]를 기존 방식으로 K씩 채워 완성.
    """

    assert prompt.ndim == 2 and prompt.size(0) == 1, "prompt must be (1, P)"
    device = model.device if hasattr(model, "device") else prompt.device

    # 준비: (1, P+G) 시퀀스 할당 및 프롬프트 고정
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt
    prompt_mask = (x != mask_id)  # 프롬프트 위치 True

    # ─────────────────────────────────────────────────────────────────────────────
    # CFG용 전방 패스: (일반 conditioned, "언컨디셔닝" 쌍) → logits 혼합
    # 주의: 언컨 템플릿은 환경에 맞게 바꾸는 게 가장 안전함. 여기선 프롬프트를 마스크로 대체.
    def forward_logits(seq):
        if cfg_scale and cfg_scale > 0.0:
            un_x = seq.clone()
            un_x[prompt_mask] = mask_id
            x_cat = torch.cat([seq, un_x], dim=0)             # (2, T)
            logits = model(x_cat).logits                       # (2, T, V)
            cond, uncond = torch.chunk(logits, 2, dim=0)      # (1,T,V) each
            return uncond + (cfg_scale + 1.0) * (cond - uncond)
        else:
            return model(seq).logits                           # (1, T, V)

    # ─────────────────────────────────────────────────────────────────────────────
    # A) 초기 채우기: init_unmask_ratio * gen_length 만큼 K씩 언마스크
    num_initial = int(gen_length * init_unmask_ratio)
    assert gen_length % max(unmask_k, 1) == 0, "gen_length must be divisible by unmask_k"
    assert num_initial % max(unmask_k, 1) == 0, "init_unmask_ratio*gen_length must be divisible by unmask_k"

    for _ in range(num_initial // unmask_k):
        mask_pos = (x == mask_id)
        if not mask_pos.any():
            break

        logits = forward_logits(x)
        logits_noisy = _add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_noisy, dim=-1)                             # (1, T)
        p = F.softmax(logits, dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (1, T)

        # [MASK] 중 확신 top-K 선택
        conf_masked = torch.where(mask_pos, x0_p, torch.full_like(x0_p, -float("inf")))
        _, topk_idx = torch.topk(conf_masked, k=unmask_k, dim=-1)
        topk_idx = topk_idx[0]  # 안전 인덱싱
        x[0, topk_idx] = x0[0, topk_idx]

    # ─────────────────────────────────────────────────────────────────────────────
    # B) Loop 단계: 매 스텝 저신뢰 K개 리마스크 → K개 재언마스크 (α-고정 유사)
    for _ in range(loop_steps):
        # 1) 현재 생성 구간(프롬프트 제외) 중에서 confidence 최하위 K 개 리마스크
        gen_pos = (x != mask_id) & (~prompt_mask)
        num_unmasked_gen = int(gen_pos.sum().item())
        if num_unmasked_gen <= 0:
            continue
        k_r = min(unmask_k, num_unmasked_gen)

        logits = forward_logits(x)
        p = F.softmax(logits, dim=-1)
        cur_token_p = torch.gather(p, dim=-1, index=x.unsqueeze(-1)).squeeze(-1)  # (1, T)
        # 생성 토큰의 현재 확률(=confidence). 값이 낮을수록 나쁨.
        conf_unmasked = torch.where(gen_pos, cur_token_p, torch.full_like(cur_token_p, float("inf")))
        _, lowk_idx = torch.topk(conf_unmasked, k=k_r, dim=-1, largest=False)
        lowk_idx = lowk_idx[0]
        x[0, lowk_idx] = mask_id  # 리마스크

        # 2) 바로 K개 재언마스크: 방금 마스크된 자리를 우선 포함하지만,
        #    일반적으로는 전체 [MASK] 중 top-K를 뽑아 같은 규모로 채움
        mask_pos = (x == mask_id)
        if mask_pos.any():
            logits = forward_logits(x)
            logits_noisy = _add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_noisy, dim=-1)
            p_new = F.softmax(logits, dim=-1)
            x0_p_new = torch.gather(p_new, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            conf_masked = torch.where(mask_pos, x0_p_new, torch.full_like(x0_p_new, -float("inf")))
            _, upk_idx = torch.topk(conf_masked, k=k_r, dim=-1)
            upk_idx = upk_idx[0]
            x[0, upk_idx] = x0[0, upk_idx]

    # ─────────────────────────────────────────────────────────────────────────────
    # C) 마무리: 남은 [MASK]를 K씩 채우기
    while (x == mask_id).any():
        mask_pos = (x == mask_id)
        k_f = min(unmask_k, int(mask_pos.sum().item()))
        if k_f <= 0:
            break

        logits = forward_logits(x)
        logits_noisy = _add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_noisy, dim=-1)
        p = F.softmax(logits, dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        conf_masked = torch.where(mask_pos, x0_p, torch.full_like(x0_p, -float("inf")))
        _, topk_idx = torch.topk(conf_masked, k=k_f, dim=-1)
        topk_idx = topk_idx[0]
        x[0, topk_idx] = x0[0, topk_idx]

    return x  # (1, P+G)

def _sample_categorical(probs: torch.Tensor) -> torch.Tensor:
    # Gumbel-Max trick; probs: (B, L, V)
    g = -torch.log(-torch.log(torch.rand_like(probs) + 1e-10) + 1e-10)
    return (probs.clamp_min(1e-12).log() + g).argmax(dim=-1)

# @torch.no_grad()
# def generate_with_remdm(
#     model,
#     prompt,
#     gen_length=32,
#     init_unmask_ratio=0.875,   # 28/32
#     unmask_k=1,                # kept for API compatibility (unused in ReMDM-conf)
#     loop_steps=32,
#     temperature=0.0,           # per-position sampling temp (set >0 if you want more randomness)
#     cfg_scale=0.0,
#     remasking='low_confidence',# kept for API compatibility (unused in ReMDM-conf)
#     mask_id=126336,
#     tokenizer=None,
# ):
#     """
#     ReMDM-conf (정식 전이식) + LLaDA 초기/마무리 하이브리드 샘플러
#       1) 초기: LLaDA/MaskGiT 식으로 init_unmask_ratio*gen_length만큼 채우기
#       2) 루프: ReMDM-conf 전이(q¹/q²)로 loop_steps 스텝 진행 (토큰별 σ = softmax(conf)*σ_max)
#       3) 잔여 [MASK]: 다시 LLaDA로 마무리
#     """
#     device = model.device if hasattr(model, "device") else prompt.device
#     B = 1
#     P = prompt.shape[1]
#     L = gen_length

#     # 확률/엔트로피 계산은 float32로 통일 (bf16 로짓을 f32로 올려 안정화)
#     fp = torch.float32
#     neg_inf_fp = torch.tensor(-float('inf'), device=device, dtype=fp)

#     # 시퀀스 준비
#     x = torch.full((B, P + L), mask_id, dtype=torch.long, device=device)
#     x[:, :P] = prompt.clone()
#     prompt_mask = (x != mask_id)

#     # ---------- 1) 초기 28/32: LLaDA/MaskGiT 진행형 채우기 ----------
#     init_tokens = int(L * init_unmask_ratio)
#     tokens_to_fill = init_tokens
#     while tokens_to_fill > 0 and (x == mask_id).any():
#         # CFG (옵션)
#         if cfg_scale > 0.:
#             un_x = x.clone()
#             un_x[prompt_mask] = mask_id
#             x_ = torch.cat([x, un_x], dim=0)
#             logits = model(x_).logits.to(fp)
#             logits, un_logits = torch.chunk(logits, 2, dim=0)
#             logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
#         else:
#             logits = model(x).logits.to(fp)  # (B, P+L, V)

#         if temperature > 0.0:
#             logits = logits / max(1e-6, temperature)

#         probs = F.softmax(logits, dim=-1)  # fp32
#         mask_pos = (x == mask_id) & (~prompt_mask)
#         if not mask_pos.any():
#             break

#         # 위치별 후보/확신
#         cand = probs.argmax(dim=-1)  # (B, P+L)
#         conf = torch.gather(probs, -1, cand.unsqueeze(-1)).squeeze(-1)  # (B, P+L) fp32
#         conf = torch.where(mask_pos, conf, neg_inf_fp)

#         # 한 번에 너무 많이 채우지 않도록 소분할
#         k = min(max(1, (tokens_to_fill + 3) // 4), tokens_to_fill)
#         _, idx = torch.topk(conf, k=k, dim=1)
#         x[torch.arange(B)[:, None], idx] = cand[torch.arange(B)[:, None], idx]
#         tokens_to_fill -= k

#     # ---------- 2) ReMDM-conf 루프: 정식 q¹/q² 전이 ----------
#     # conf 캐시: 논문식 - 마지막 언마스크 시점의 디코딩 확률(부호 반대 저장). fp32로 유지
#     conf_cache = torch.full_like(x, fill_value=-float('inf'), dtype=fp)

#     eps = 1e-5
#     timesteps = torch.linspace(1.0, eps, steps=loop_steps + 1, device=device)  # 1 → eps
#     dt = (1.0 - eps) / max(1, loop_steps)

#     for i in range(loop_steps):
#         t = timesteps[i].item()
#         t_s = max(eps, t - dt)

#         # logits → p_x0
#         if cfg_scale > 0.:
#             un_x = x.clone()
#             un_x[prompt_mask] = mask_id
#             x_ = torch.cat([x, un_x], dim=0)
#             logits = model(x_).logits.to(fp)
#             logits, un_logits = torch.chunk(logits, 2, dim=0)
#             logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
#         else:
#             logits = model(x).logits.to(fp)

#         if temperature > 0.0:
#             logits = logits / max(1e-6, temperature)

#         p_x0 = F.softmax(logits, dim=-1)  # (B, P+L, V), fp32

#         # α_t, α_s, σ_max
#         alpha_t = max(0.0, 1.0 - t)
#         alpha_s = max(0.0, 1.0 - t_s)
#         denom = max(1e-12, alpha_t)
#         sigma_max = min(1.0, (1.0 - alpha_s) / denom)

#         # η = softmax(conf_cache)  (마스크 위치는 0)
#         eta = torch.softmax(conf_cache, dim=-1)
#         eta = torch.where((x == mask_id), torch.zeros_like(eta, dtype=fp), eta)

#         # σ (토큰별): σ = η * σ_max
#         sigma = eta * sigma_max  # (B, P+L), fp32

#         # 전이 분포 q¹(언마스크용), q²(마스크용)
#         q1 = p_x0 * (1.0 - sigma[:, :, None])
#         q1[..., mask_id] = sigma

#         num = alpha_s - (1.0 - sigma) * alpha_t
#         den = max(1e-12, (1.0 - alpha_t))
#         q2 = p_x0 * (num[:, :, None] / den)
#         q2[..., mask_id] = (1.0 - alpha_s - sigma * alpha_t) / den

#         is_unmasked = (x != mask_id)
#         q = torch.where(is_unmasked.unsqueeze(-1), q1, q2)
#         q = q.clamp_min(0)
#         q = q / (q.sum(dim=-1, keepdim=True) + 1e-12)

#         # 카테고리 샘플링
#         xs = _sample_categorical(q)  # (B, P+L)

#         # conf 갱신 (논문식):
#         #  - 마스크→언마스크: conf = -p_x0[선택토큰]
#         #  - 언마스크→마스크: conf = -inf
#         became_unmasked = (x == mask_id) & (xs != mask_id)
#         became_masked   = (x != mask_id) & (xs == mask_id)
#         chosen_prob = torch.gather(p_x0, -1, xs.unsqueeze(-1)).squeeze(-1)  # (B, P+L), fp32

#         conf_cache = conf_cache.clone()
#         conf_cache[became_unmasked] = (-chosen_prob)[became_unmasked].to(conf_cache.dtype)
#         conf_cache[became_masked]   = neg_inf_fp

#         x = xs

#     # ---------- 3) 잔여 [MASK]는 LLaDA/MaskGiT로 마무리 ----------
#     while (x == mask_id).any():
#         if cfg_scale > 0.:
#             un_x = x.clone()
#             un_x[prompt_mask] = mask_id
#             x_ = torch.cat([x, un_x], dim=0)
#             logits = model(x_).logits.to(fp)
#             logits, un_logits = torch.chunk(logits, 2, dim=0)
#             logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
#         else:
#             logits = model(x).logits.to(fp)

#         if temperature > 0.0:
#             logits = logits / max(1e-6, temperature)

#         p = F.softmax(logits, dim=-1)  # fp32
#         cand = _sample_categorical(p)
#         mask_pos = (x == mask_id)
#         x[mask_pos] = cand[mask_pos]

#     # 옵션: 디코드
#     if tokenizer is not None:
#         _ = tokenizer.decode(x[0, P:], skip_special_tokens=False)

#     return x
