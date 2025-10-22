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
@torch.no_grad()
def generate_with_refine_ent3(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking='low_confidence',
    mask_id=126336,
    refine_every=1,
    nucleus_p=1.0,
    return_order=False,
):
    """
    LLaDA용 refine-ent-3 (B-type) 샘플러:
      1) 각 스텝에서 기본 전이로 일부 토큰 언마스크
      2) 주기(refine_every)마다 R 위치를 선택해 2nd-forward로 즉시 refine
      3) refine 직후 분포로 해당 위치의 entropy를 재계산하여 confidence 캐시에 '덮어쓰기'
    """
    device = model.device
    B = 1
    x = torch.full((B, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if return_order:
        orders = {}

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    # dtype을 float32로 통일해 확률/엔트로피 계산의 안정성 확보
    fp = torch.float32
    eps = torch.finfo(fp).eps

    # confidence 캐시 (entropy). 초기에는 매우 낮게 설정
    conf = torch.full_like(x, fill_value=-1e9, dtype=fp)

    step_counter = 0

    for num_block in range(num_blocks):
        blk_lo = prompt.shape[1] + num_block * block_length
        blk_hi = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, blk_lo:blk_hi] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = (x == mask_id)

            # predictor
            logits = model(x).logits.to(fp)  # float32 변환
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L)

            # confidence(=1-entropy 아님! 아래에서 entropy 계산 전까진 top1 확률로 사용)
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)  # fp32
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1).to(fp)  # (B, L)
            elif remasking == 'random':
                x0_p = torch.rand_like(x0, dtype=fp)
            else:
                raise NotImplementedError(remasking)

            # 블록 밖은 아직 잠금
            x0_p[:, blk_hi:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=device, dtype=fp))

            # 해당 스텝에서 확정할 개수만 선택
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
            for b in range(B):
                k = int(num_transfer_tokens[b, i].item())
                if k > 0:
                    _, sel = torch.topk(confidence[b], k=k)
                    transfer_index[b, sel] = True
                    if return_order:
                        blk = num_block + 1
                        if blk not in orders:
                            orders[blk] = []
                        orders[blk].append((sel - prompt.shape[1]).tolist())
            x[transfer_index] = x0[transfer_index]

            # --- (B) in-step 2-forward refine (refine_every 간격) ---
            if (step_counter % max(1, refine_every)) == 0:
                # 1) 확정된 위치들 중 [MASK]가 아닌 곳만 대상으로 entropy 기반 선택 가중치
                with torch.no_grad():
                    p1 = F.softmax(logits, dim=-1).clamp_min(eps)  # fp32
                    p1_wo = p1.clone()
                    p1_wo[..., mask_id] = 0.0
                    Z = p1_wo.sum(dim=-1, keepdim=True).clamp_min(eps)
                    q = p1_wo / Z
                    H = -(q * (q + eps).log()).sum(dim=-1)  # (B, L) entropy, fp32

                # [MASK]였던 곳 제외
                unmasked_flag = (x != mask_id)
                eta = torch.softmax(H, dim=-1).masked_fill(~unmasked_flag, 0.0)  # 확정 위치만 후보
                # sigma = eta * sigma_max → LLaDA에는 명시적 schedule이 없으므로 sigma_max=1.0 사용
                sigma = eta.clamp(0.0, 1.0)
                R = (torch.rand_like(sigma) < sigma) & unmasked_flag
                if R.any():
                    # 2) R만 remask하여 2nd forward
                    x_tmp = x.clone()
                    x_tmp[R] = mask_id
                    logits2 = model(x_tmp).logits.to(fp)  # fp32

                    # nucleus(top-p) 옵션
                    if nucleus_p < 1.0:
                        p2 = F.softmax(logits2, dim=-1)  # fp32
                        sorted_probs, sorted_idx = torch.sort(p2, dim=-1, descending=True)
                        cprob = torch.cumsum(sorted_probs, dim=-1)
                        keep = (cprob <= nucleus_p)
                        keep[..., 0] = True
                        nucleus = sorted_probs * keep
                        nucleus = nucleus / nucleus.sum(dim=-1, keepdim=True).clamp_min(eps)
                        p_x0_2 = torch.zeros_like(p2).scatter_(-1, sorted_idx, nucleus)
                    else:
                        p_x0_2 = F.softmax(logits2, dim=-1)  # fp32

                    # 3) R 위치만 재샘플 → 즉시 덮어쓰기
                    sampled2 = torch.argmax(
                        add_gumbel_noise(torch.log(p_x0_2.clamp_min(eps)), temperature=0.0),
                        dim=-1
                    )
                    x[R] = sampled2[R]

                    # 4) (핵심) refine 직후 분포의 entropy로 conf 캐시 덮어쓰기
                    P2 = p_x0_2.clamp_min(eps)  # fp32
                    P2[..., mask_id] = 0.0
                    Z2 = P2.sum(dim=-1, keepdim=True).clamp_min(eps)
                    Q2 = P2 / Z2
                    H2 = -(Q2 * (Q2 + eps).log()).sum(dim=-1)
                    conf[R] = H2[R].to(conf.dtype)

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

def _confidence_from_logits(logits, x0, mask_index, mask_id, metric="prob"):
    """
    logits: (B, L, V)
    x0:     (B, L)  argmax 토큰
    mask_index: (B, L)  현재 샘플링 대상으로 삼을 위치(True)
    metric: "prob" | "entropy"
    반환: (B, L)  값이 클수록 더 빨리 확정할 위치
    """
    fp = torch.float32
    eps = torch.finfo(fp).eps
    device = logits.device

    if metric == "prob":
        p = F.softmax(logits.to(fp), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1).to(fp)
        conf = torch.where(mask_index, x0_p, torch.full_like(x0_p, -float("inf")))
        return conf

    if metric == "entropy":
        p = F.softmax(logits.to(fp), dim=-1).clamp_min(eps)
        # [MASK] 토큰 확률을 제거한 뒤 재정규화 (언마스크 후보의 불확실성만 반영)
        p_wo = p.clone()
        p_wo[..., mask_id] = 0.0
        Z = p_wo.sum(dim=-1, keepdim=True).clamp_min(eps)
        q = p_wo / Z
        H = -(q * (q + eps).log()).sum(dim=-1)  # (B, L)
        score = -H  # 값이 클수록 더 확정할만함
        conf = torch.where(mask_index, score, torch.full_like(score, -float("inf")))
        return conf

    raise ValueError(f"Unknown confidence_metric: {metric}")

# @torch.no_grad()
# def generate_with_remdm(model, prompt, gen_length=32, init_unmask_ratio=0.875, unmask_k=1, loop_steps=32, temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336, tokenizer=None):
#     assert 0.0 <= init_unmask_ratio <= 1.0, "init_unmask_ratio must be between 0 and 1"
#     num_initial_tokens = gen_length * init_unmask_ratio
#     assert num_initial_tokens == int(num_initial_tokens), "gen_length * init_unmask_ratio must be an integer"
#     num_initial_tokens = int(num_initial_tokens)
#     assert gen_length % unmask_k == 0, "gen_length must be divisible by unmask_k"
#     assert num_initial_tokens % unmask_k == 0, "init_unmask_ratio * gen_length must be divisible by unmask_k"
#     x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
#     x[:, :prompt.shape[1]] = prompt.clone()
#     prompt_index = (x != mask_id)
#     num_loops = num_initial_tokens // unmask_k
#     for _ in range(num_loops):
#         mask_index = (x == mask_id)
#         if not mask_index.any():
#             break

#         if cfg_scale > 0.:
#             un_x = x.clone()
#             un_x[prompt_index] = mask_id
#             x_ = torch.cat([x, un_x], dim=0)
#             logits = model(x_).logits
#             logits, un_logits = torch.chunk(logits, 2, dim=0)
#             logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
#         else:
#             logits = model(x).logits
#         logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
#         x0 = torch.argmax(logits_with_noise, dim=-1)
#         p = F.softmax(logits, dim=-1)
#         x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
#         confidence = torch.where(mask_index, x0_p, -torch.inf)
#         _, select_indices = torch.topk(confidence, k=unmask_k)
#         x[0, select_indices] = x0[0, select_indices]
#         x_result = tokenizer.decode(x[0, prompt.shape[1]:], skip_special_tokens=False)
#     for _ in range(loop_steps):
#         unmasked_gen_index = (x != mask_id) & (~prompt_index)
#         num_unmasked_gen = torch.sum(unmasked_gen_index).item()
#         if num_unmasked_gen == 0:
#             continue
#         current_remask_k = min(unmask_k, num_unmasked_gen)
#         if cfg_scale > 0.:
#             un_x = x.clone()
#             un_x[prompt_index] = mask_id
#             x_ = torch.cat([x, un_x], dim=0)
#             logits = model(x_).logits
#             logits, un_logits = torch.chunk(logits, 2, dim=0)
#             logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
#         else:
#             logits = model(x).logits
#         p = F.softmax(logits, dim=-1)
#         current_token_p = torch.gather(p, dim=-1, index=x.unsqueeze(-1)).squeeze(-1)
#         confidence = torch.where(unmasked_gen_index, current_token_p, torch.inf)
#         _, remask_indices = torch.topk(confidence, k=current_remask_k, largest=False)
#         x[0, remask_indices] = mask_id
#         mask_index_after_remask = (x == mask_id)
#         if cfg_scale > 0.:
#             un_x = x.clone()
#             un_x[prompt_index] = mask_id
#             x_ = torch.cat([x, un_x], dim=0)
#             logits = model(x_).logits
#             logits, un_logits = torch.chunk(logits, 2, dim=0)
#             logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
#         else:
#             logits = model(x).logits
#         logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
#         x0 = torch.argmax(logits_with_noise, dim=-1)
#         p_new = F.softmax(logits, dim=-1)
#         x0_p_new = torch.gather(p_new, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
#         confidence_for_unmasking = torch.where(mask_index_after_remask, x0_p_new, -torch.inf)
#         _, unmask_indices = torch.topk(confidence_for_unmasking, k=current_remask_k)
#         x[0, unmask_indices] = x0[0, unmask_indices]
#         x_result = tokenizer.decode(x[0, prompt.shape[1]:], skip_special_tokens=False)
#     while (x == mask_id).any():
#         mask_index = (x == mask_id)
#         num_masked_left = torch.sum(mask_index).item()
#         if num_masked_left == 0:
#             break
#         current_unmask_k = min(unmask_k, num_masked_left)
#         if cfg_scale > 0.:
#             un_x = x.clone()
#             un_x[prompt_index] = mask_id
#             x_ = torch.cat([x, un_x], dim=0)
#             logits = model(x_).logits
#             logits, un_logits = torch.chunk(logits, 2, dim=0)
#             logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
#         else:
#             logits = model(x).logits
#         logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
#         x0 = torch.argmax(logits_with_noise, dim=-1)
#         p = F.softmax(logits, dim=-1)
#         x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
#         confidence = torch.where(mask_index, x0_p, -torch.inf)
#         _, select_indices = torch.topk(confidence, k=current_unmask_k)
#         x[0, select_indices] = x0[0, select_indices]
#         x_result = tokenizer.decode(x[0, prompt.shape[1]:], skip_special_tokens=False)
#     return x

import torch
import torch.nn.functional as F

def _sample_categorical(probs: torch.Tensor) -> torch.Tensor:
    # Gumbel-Max trick; probs: (B, L, V)
    g = -torch.log(-torch.log(torch.rand_like(probs) + 1e-10) + 1e-10)
    return (probs.clamp_min(1e-12).log() + g).argmax(dim=-1)

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
    ReMDM-conf (정식 전이식) + LLaDA 초기/마무리 하이브리드 샘플러
      1) 초기: LLaDA/MaskGiT 식으로 init_unmask_ratio*gen_length만큼 채우기
      2) 루프: ReMDM-conf 전이(q¹/q²)로 loop_steps 스텝 진행 (토큰별 σ = softmax(conf)*σ_max)
      3) 잔여 [MASK]: 다시 LLaDA로 마무리
    """
    device = model.device if hasattr(model, "device") else prompt.device
    B = 1
    P = prompt.shape[1]
    L = gen_length

    # 시퀀스 준비
    x = torch.full((B, P + L), mask_id, dtype=torch.long, device=device)
    x[:, :P] = prompt.clone()
    prompt_mask = (x != mask_id)

    # ---------- 1) 초기 28/32: LLaDA/MaskGiT 진행형 채우기 ----------
    init_tokens = int(L * init_unmask_ratio)
    tokens_to_fill = init_tokens
    while tokens_to_fill > 0 and (x == mask_id).any():
        # CFG (옵션)
        if cfg_scale > 0.:
            un_x = x.clone()
            un_x[prompt_mask] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            logits = model(x_).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(x).logits  # (B, P+L, V)

        if temperature > 0.0:
            logits = logits / max(1e-6, temperature)

        probs = F.softmax(logits, dim=-1)
        mask_pos = (x == mask_id) & (~prompt_mask)
        if not mask_pos.any():
            break

        # 위치별 후보/확신
        cand = probs.argmax(dim=-1)  # (B, P+L)
        conf = torch.gather(probs, -1, cand.unsqueeze(-1)).squeeze(-1)
        conf = torch.where(mask_pos, conf, torch.tensor(-float('inf'), device=device))

        # 한 번에 너무 많이 채우지 않도록 소분할
        k = min(max(1, (tokens_to_fill + 3) // 4), tokens_to_fill)
        _, idx = torch.topk(conf, k=k, dim=1)
        x[torch.arange(B)[:, None], idx] = cand[torch.arange(B)[:, None], idx]
        tokens_to_fill -= k

    # ---------- 2) ReMDM-conf 루프: 정식 q¹/q² 전이 ----------
    # conf 캐시: 논문식 - 마지막 언마스크 시점의 디코딩 확률(부호 반대 저장)
    conf_cache = torch.full_like(x, fill_value=-float('inf'), dtype=torch.float32)

    eps = 1e-5
    timesteps = torch.linspace(1.0, eps, steps=loop_steps + 1, device=device)  # 1 → eps
    dt = (1.0 - eps) / max(1, loop_steps)

    for i in range(loop_steps):
        t = timesteps[i].item()
        t_s = max(eps, t - dt)

        # logits → p_x0
        if cfg_scale > 0.:
            un_x = x.clone()
            un_x[prompt_mask] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            logits = model(x_).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(x).logits

        if temperature > 0.0:
            logits = logits / max(1e-6, temperature)

        p_x0 = F.softmax(logits, dim=-1)  # (B, P+L, V)

        # α_t, α_s, σ_max
        alpha_t = max(0.0, 1.0 - t)
        alpha_s = max(0.0, 1.0 - t_s)
        denom = max(1e-12, alpha_t)
        sigma_max = min(1.0, (1.0 - alpha_s) / denom)

        # η = softmax(conf_cache)  (마스크 위치는 0)
        eta = torch.softmax(conf_cache, dim=-1)
        eta = torch.where((x == mask_id), torch.zeros_like(eta), eta)

        # σ (토큰별): σ = η * σ_max
        sigma = eta * sigma_max  # (B, P+L)

        # 전이 분포 q¹(언마스크용), q²(마스크용)
        q1 = p_x0 * (1.0 - sigma[:, :, None])
        q1[..., mask_id] = sigma

        num = alpha_s - (1.0 - sigma) * alpha_t
        den = max(1e-12, (1.0 - alpha_t))
        q2 = p_x0 * (num[:, :, None] / den)
        q2[..., mask_id] = (1.0 - alpha_s - sigma * alpha_t) / den

        is_unmasked = (x != mask_id)
        q = torch.where(is_unmasked.unsqueeze(-1), q1, q2)
        q = q.clamp_min(0)
        q = q / (q.sum(dim=-1, keepdim=True) + 1e-12)

        # 카테고리 샘플링
        xs = _sample_categorical(q)  # (B, P+L)

        # conf 갱신 (논문식):
        #  - 마스크→언마스크: conf = -p_x0[선택토큰]
        #  - 언마스크→마스크: conf = -inf
        became_unmasked = (x == mask_id) & (xs != mask_id)
        became_masked   = (x != mask_id) & (xs == mask_id)
        chosen_prob = torch.gather(p_x0, -1, xs.unsqueeze(-1)).squeeze(-1)  # (B, P+L)

        conf_cache = conf_cache.clone()
        conf_cache[became_unmasked] = (-chosen_prob)[became_unmasked]
        conf_cache[became_masked]   = -float('inf')

        x = xs

    # ---------- 3) 잔여 [MASK]는 LLaDA/MaskGiT로 마무리 ----------
    while (x == mask_id).any():
        if cfg_scale > 0.:
            un_x = x.clone()
            un_x[prompt_mask] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            logits = model(x_).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(x).logits

        if temperature > 0.0:
            logits = logits / max(1e-6, temperature)

        p = F.softmax(logits, dim=-1)
        cand = _sample_categorical(p)
        mask_pos = (x == mask_id)
        x[mask_pos] = cand[mask_pos]

    # 옵션: 디코드
    if tokenizer is not None:
        _ = tokenizer.decode(x[0, P:], skip_special_tokens=False)

    return x
