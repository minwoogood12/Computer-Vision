# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging #로깅 라이브러리 

import torch
import torch.nn.functional as F #loss함수, 연산함수가 들어있는 모듈 ex. cross_entropy, unfold
from torch import nn #nn.Module을 위함

from detectron2.utils.comm import get_world_size #Detectron2에서 총 프로세스 수 가져오기
from detectron2.projects.point_rend.point_features import ( 
    get_uncertain_point_coords_with_randomness, #마스크 예측  중 가장 불확실한 지점 선택
    point_sample, # 선택된 포인트 위치에서 feature map값을 샘플링 
) # => supervision 시 point-wise loss를 계산, 즉 전체가 아닌 중요한 부분만 뽑아서 학습

#feature map-> convolution 1번 (디코더) -> mask logit(차이가 큰 점수) -> [sigmoid or logsigmoid]하면 마스크확률 x(dice_coefficient)



from mask2former.utils.misc import is_dist_avail_and_initialized #detectron 기반 마스크2포머에서 분산 학습 여부 확인 

def unfold_wo_center(x, kernel_size, dilation): #중심 제거하고 주변 픽셀 얻기
    assert x.dim() == 4 # x는 4차원 텐서 [B x C x H x W]
    assert kernel_size % 2 == 1 # 커널 크기가 홀수이여야함 -> 중심 픽셀이 있어야 중심만 제거 가능 

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2 #컨볼루션 출력을 입력과 같게하기 위해 패딩을 임의로 설정
    unfolded_x = F.unfold( #이미지에 슬라이딩윈도우를 적용
        x, kernel_size=kernel_size, #자르려는 커널 크기
        padding=padding, #입력에 0패딩 붙여서 크게만듬
        dilation=dilation #커널을 n칸씩 띄움
    )
    # => 출력 [B, C * kernel_size^2 , H x W ] 4에서 3차원으로 바뀜 왜냐면 H , W가 합쳐지고 H * W으로 차원을 축소시킴
    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    ) # 다시 [B, C, K^2(자동), H, W] 로 구조화함

    # remove the center pixels #커널 중심 제거 
    size = kernel_size ** 2 #커널 사이즈 구함
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2) #dim = 2 즉 kernel차원에서 해당하는 중심을 제외한 나머지를 앞뒤로 이어붙임

    return unfolded_x 
    #입력 [B x C x H x W] 4차원 텐서 
    #출력은 [B, C, k^2, H , W] 5차원 텐서
    #즉 주변의 정보를 담아서 리턴

def unfold_w_center(x, kernel_size, dilation): #중심 포함 주변 픽셀 얻기 
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    return unfolded_x

#L_pair 구하는과정 (공간)

def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    #pairwise_size : 커널 크기랑 비슷 pairwise_dilation : 위의 딜레이션이랑 비슷
    #현재 프레임에서 주변 픽셀들과의 로스를 구함
    assert mask_logits.dim() == 4 #디코더의 conv 마스크 예측 값 [B, 1, H, W] 형태

    # 차이가 큰 마스크 로짓 점수를 -> 시그모이드 + log로 안정화
    log_fg_prob = F.logsigmoid(mask_logits) # 전경 4차원 텐서
    log_bg_prob = F.logsigmoid(-mask_logits) # 후경 4차원 텐서
    #시그모이드가 0~1사이로 바꿔주는거
    #근데 시그모이드만 쓰면 나중에 로스구할때 숫자가 불안정해짐 log(0.0000001)
    #그래서 2 ->시그모이드 0.8 -> 로그 -0.12
  
    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    # 전경 마스크주변값들 뽑아냄 (중심 제외) 5차원 텐서로 바뀜 중심 제거하는이유는 : 자기 자신이랑 비교하기 때문에
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
     # 후경 마스크 주변값들 뽑아냄 (중심 제외) 5차원 텐서로 바뀜
    
    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    # Lcons 구하는 과정 
    # 예측된 마스크와 인접 픽셀 더하기, 이건 로그형식이라 원래는 곱하긴데 더하기가 가능
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold #전경
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold #후경
    #[ : , :, None] => 4차원 텐서를 5차원텐서로 변환(Broadcast)qm, kernel자리를 추가 하지만 0이나 의미없는 값추가가 아니라 공간만 만들어내는거 ??
  
    max_ = torch.max(log_same_fg_prob, log_same_bg_prob) #전경 후경중 높은 값 선택 => 오버플로우 방지용 (로그때문)
  
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_
    #위 코드는 아무튼 간에 전경과 후경을 로그를 통해서 log (전경 + 후경)하게 해줌 돌아간 방식임
  
    # loss = -log(prob) #로스는 최종적으로 -해줘야함
    return -log_same_prob[:, 0] #채널이 0이기때문에 없애도됨 => 계산편의성을 위해서
    #입력 [B, 1, H, W] 4차원 마스크 예측 로짓
    #출력 [B, K^2, H, W] 4차원 로그 값 (L_cons)
    #즉 공간 로스를 구하는 과정 (L_cons)

def compute_pairwise_term_neighbor(mask_logits, mask_logits_neighbor, pairwise_size, pairwise_dilation):
    #이웃프레임 마스크와 현재프레임의 주변픽셀들과 비교
    assert mask_logits.dim() == 4 #디코더의 conv 마스크 예측 값 [B, 1, H, W] 형태

    log_fg_prob_neigh = F.logsigmoid(mask_logits_neighbor)
    log_bg_prob_neigh = F.logsigmoid(-mask_logits_neighbor)
    #이웃 프레임의 점수를 전경과 후경 둘다 시그모이드+로그로 안정화
  
    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)
    #현재 프레임의 점수를 전경과 후경 둘다 시그모이드+로그로 안정화 
  
    log_fg_prob_unfold = unfold_w_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    #현재 프레임의 전경의 주변 픽셀들을 구함 (중심 포함)
    log_bg_prob_unfold = unfold_w_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    #현재 프레임의 후경의 주변 픽셀들을 구함 (중심 포함) = 자기 자신과 비교 안해서, 다른 프레임과 비교해서
  
    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob_neigh[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob_neigh[:, :, None] + log_bg_prob_unfold
    #현재 프레임 전경 + 이웃 프레임 전경 / 현재 프레임 후경 + 이웃 프레임 후경


    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_
    #최종 전경과 후경을 더해줌
  
    return -log_same_prob[:, 0] #-를 해주고 커널 차원을 없애줌 (계산 간편)
    #입력 [B, 1, H, W] 4차원 마스크 예측 로짓
    #출력 [B, K^2, H, W] 4차원 로그 값 (L_cons)
    #즉 시간 로스를 구하는 과정 (L_cons)

def dice_coefficient(x, target): # D(projection된 마스크, projection된 마스크) - L_proj
    #입력 [B, 1, H, W] 둘다 이건 둘다 projection하고 다시 돌려놓은거 ex. [B, 1, H, W] -> [B, 1, H, 1] -> [B, 1, H, W] 
    #x는 마스크 예측으로 나온 마스크 로짓에 시그모이드 한것 sigmoid(mask_logits) + projection 후 다시 broadcast로 복구 
    #target은 gt box로 마스크를 만든 것 -> 이건 L_pair 구할때만 계산됨
    
    eps = 1e-5 #나눗셈에서 0나누기 방지용 ?
    n_inst = x.size(0) #입력 마스크의 개수 
    
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    #x, target_gt를 모두 [B, 1, H, W] -> 첫번째 n_inst : B배치 크기만 두고 나머지를 합치는것
    # => [B, 1 * H * W] 형태로 변환
    
    intersection = (x * target).sum(dim=1)
    #x, target간의 교집합을 구하기 A n B
    
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    # |A| + |B| 인데 제곱하고 더함으로써 0~1사이의 숫자가 더 0.8 => 0.64, 0.1 => 0.01로 더 가중치를 줄수있음 
    
    loss = 1. - (2 * intersection / union)
    #Dice coef : 유사도 즉 높을수록 좋은 예측 => 예측이 좋을 수록 로스를 적게 주기 위해서 1에서 빼줌
    return loss
    #입력 [B, 1, H, W] 4차원 텐서
    #출력 [B] 1차원 텐서 (loss값을 각각 배치마다 가지고 있음)

def compute_project_term(mask_scores, gt_bitmasks):L_proj에서 D값을 각각 x,y 두개를 더해서 평균 반환
    #mask_scores : mask_logit에 시그모이드 함수 처리 [B, 1, H, W]
    #gt_bitmasks : GT Box 기반으로 만든 이진 마스크
    
    mask_losses_y = dice_coefficient(
        mask_scores.max(dim=2, keepdim=True)[0],
        gt_bitmasks.max(dim=2, keepdim=True)[0]
    )
    #예측 마스크와 gt마스크를 y차원으로 prediction해서 유사도 비교
    
    mask_losses_x = dice_coefficient(
        mask_scores.max(dim=3, keepdim=True)[0],
        gt_bitmasks.max(dim=3, keepdim=True)[0]
    )
    #예측 마스크와 gt마스크를 x차원으로 prediction해서 유사도 비교 
    # [0.2, 0.1, 0.3, 0.4]형태 1차원 텐서
    
    return (mask_losses_x + mask_losses_y).mean() #각각 방향의 dice loss의 평균을 구해 반환
    #1차원 두개를 더하고 그 안에서 평균을 내기때문에 실수 형태로 리턴
    #입력 [B, 1, H, W]
    #출력 하나의 실수 형태(스칼라)

def dice_loss(  #마스크 GT 있을때만  # 마스크예측과 GT마스크 간의 겹치는 정도 #높을수록 유사도 높음
        inputs: torch.Tensor, #모델의 예측 마스크 로짓값
        targets: torch.Tensor, #정답 마스크 이진값
        num_masks: float, #마스크 수
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid() #시그모이드로 확률화
    inputs = inputs.flatten(1) #마스크를 [B, H*W]로 전처리
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    #Dice Loss 구하기
        
    loss = 1 - (numerator + 1) / (denominator + 1)
    #로스 구하기
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss( #마스크 GT 있을때만 #픽셀의 맞은지 틀린지 하나씩 비교 #작을수록 일치
        inputs: torch.Tensor, #마스크 예측 Logit [B, 1, H, W]
        targets: torch.Tensor, # [B, 1, H, W]
        num_masks: float,#마스크 개수
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    #BCE Loss계산 , 내부적으로 sigmoid 적용후 BCE Loss계산, reduction = "none" : 개별 로스를 그대로 반환 (평균 x)

        
    return loss.mean(1).sum() / num_masks #평균 loss계산해서 모든   마스크 합하고 나누기 마스크 수


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits): #마스크 GT 있을때만 # 불확실한 부분 강조 #마스크2포머의 k개 샘플링하는거
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertai locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class VideoSetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        self._warmup_iters = 2000
        self.register_buffer("_iter", torch.zeros([1]))

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses
    
   
    def topk_mask(self, images_lab_sim, k):
        images_lab_sim_mask = torch.zeros_like(images_lab_sim)
        topk, indices = torch.topk(images_lab_sim, k, dim =1) # 1, 3, 5, 7
        images_lab_sim_mask = images_lab_sim_mask.scatter(1, indices, topk)
        return images_lab_sim_mask

    def loss_masks_proj(self, outputs, targets, indices, num_masks, images_lab_sim, images_lab_sim_nei, images_lab_sim_nei1, images_lab_sim_nei2, images_lab_sim_nei3, images_lab_sim_nei4):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        
        self._iter += 1

        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        # Modified to handle video
        target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)]).to(src_masks)

        images_lab_sim = torch.cat(images_lab_sim, dim =0)
        images_lab_sim_nei = torch.cat(images_lab_sim_nei, dim=0)
        images_lab_sim_nei1 = torch.cat(images_lab_sim_nei1, dim=0)
        images_lab_sim_nei2 = torch.cat(images_lab_sim_nei2, dim=0)
        images_lab_sim_nei3 = torch.cat(images_lab_sim_nei3, dim=0)
        images_lab_sim_nei4 = torch.cat(images_lab_sim_nei4, dim=0)

        images_lab_sim = images_lab_sim.view(-1, target_masks.shape[1], images_lab_sim.shape[-3], images_lab_sim.shape[-2], images_lab_sim.shape[-1])
        images_lab_sim_nei = images_lab_sim_nei.unsqueeze(1)
        images_lab_sim_nei1 = images_lab_sim_nei1.unsqueeze(1)
        images_lab_sim_nei2 = images_lab_sim_nei2.unsqueeze(1)
        images_lab_sim_nei3 = images_lab_sim_nei3.unsqueeze(1)
        images_lab_sim_nei4 = images_lab_sim_nei4.unsqueeze(1)

        if len(src_idx[0].tolist()) > 0:
            images_lab_sim = torch.cat([images_lab_sim[ind][None] for ind in src_idx[0].tolist()]).flatten(0, 1)
            images_lab_sim_nei = self.topk_mask(torch.cat([images_lab_sim_nei[ind][None] for ind in src_idx[0].tolist()]).flatten(0, 1), 5)
            images_lab_sim_nei1 = self.topk_mask(torch.cat([images_lab_sim_nei1[ind][None] for ind in src_idx[0].tolist()]).flatten(0, 1), 5)
            images_lab_sim_nei2 = self.topk_mask(torch.cat([images_lab_sim_nei2[ind][None] for ind in src_idx[0].tolist()]).flatten(0, 1), 5)
            images_lab_sim_nei3 = self.topk_mask(torch.cat([images_lab_sim_nei3[ind][None] for ind in src_idx[0].tolist()]).flatten(0, 1), 5)
            images_lab_sim_nei4 = self.topk_mask(torch.cat([images_lab_sim_nei4[ind][None] for ind in src_idx[0].tolist()]).flatten(0, 1), 5)

        k_size = 3 

        if src_masks.shape[0] > 0:
            pairwise_losses_neighbor = compute_pairwise_term_neighbor(
                src_masks[:,:1], src_masks[:,1:2], k_size, 3
            ) 
            pairwise_losses_neighbor1 = compute_pairwise_term_neighbor(
                src_masks[:,1:2], src_masks[:,2:3], k_size, 3
            ) 
            pairwise_losses_neighbor2 = compute_pairwise_term_neighbor(
                src_masks[:,2:3], src_masks[:,3:4], k_size, 3
            )
            pairwise_losses_neighbor3 = compute_pairwise_term_neighbor(
                src_masks[:,3:4], src_masks[:,4:5], k_size, 3
            )
            pairwise_losses_neighbor4 = compute_pairwise_term_neighbor(
                src_masks[:,4:5], src_masks[:,0:1], k_size, 3
            )
            
        # print('pairwise_losses_neighbor:', pairwise_losses_neighbor.shape)
        src_masks = src_masks.flatten(0, 1)[:, None]
        target_masks = target_masks.flatten(0, 1)[:, None]
        target_masks = F.interpolate(target_masks, (src_masks.shape[-2], src_masks.shape[-1]), mode='bilinear')
        # images_lab_sim = F.interpolate(images_lab_sim, (src_masks.shape[-2], src_masks.shape[-1]), mode='bilinear')
        
        
        if src_masks.shape[0] > 0:
            loss_prj_term = compute_project_term(src_masks.sigmoid(), target_masks)  

            pairwise_losses = compute_pairwise_term(
                src_masks, k_size, 2
            )

            weights = (images_lab_sim >= 0.3).float() * target_masks.float()
            target_masks_sum = target_masks.reshape(pairwise_losses_neighbor.shape[0], 5, target_masks.shape[-2], target_masks.shape[-1]).sum(dim=1, keepdim=True)
            
            target_masks_sum = (target_masks_sum >= 1.0).float() # ori is 1.0
            weights_neighbor = (images_lab_sim_nei >= 0.05).float() * target_masks_sum # ori is 0.5, 0.01, 0.001, 0.005, 0.0001, 0.02, 0.05, 0.075, 0.1 , dy 0.5
            weights_neighbor1 = (images_lab_sim_nei1 >= 0.05).float() * target_masks_sum # ori is 0.5, 0.01, 0.001, 0.005, 0.0001, 0.02, 0.05, 0.075, 0.1, dy 0.5
            weights_neighbor2 = (images_lab_sim_nei2 >= 0.05).float() * target_masks_sum # ori is 0.5, 0.01, 0.001, 0.005, 0.0001, 0.02, 0.05, 0.075, 0.1, dy 0.5
            weights_neighbor3 = (images_lab_sim_nei3 >= 0.05).float() * target_masks_sum
            weights_neighbor4 = (images_lab_sim_nei4 >= 0.05).float() * target_masks_sum

            warmup_factor = min(self._iter.item() / float(self._warmup_iters), 1.0) #1.0

            loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)
            loss_pairwise_neighbor = (pairwise_losses_neighbor * weights_neighbor).sum() / weights_neighbor.sum().clamp(min=1.0) * warmup_factor
            loss_pairwise_neighbor1 = (pairwise_losses_neighbor1 * weights_neighbor1).sum() / weights_neighbor1.sum().clamp(min=1.0) * warmup_factor
            loss_pairwise_neighbor2 = (pairwise_losses_neighbor2 * weights_neighbor2).sum() / weights_neighbor2.sum().clamp(min=1.0) * warmup_factor
            loss_pairwise_neighbor3 = (pairwise_losses_neighbor3 * weights_neighbor3).sum() / weights_neighbor3.sum().clamp(min=1.0) * warmup_factor
            loss_pairwise_neighbor4 = (pairwise_losses_neighbor4 * weights_neighbor4).sum() / weights_neighbor4.sum().clamp(min=1.0) * warmup_factor

        else:
            loss_prj_term = src_masks.sum() * 0.
            loss_pairwise = src_masks.sum() * 0.
            loss_pairwise_neighbor = src_masks.sum() * 0.
            loss_pairwise_neighbor1 = src_masks.sum() * 0.
            loss_pairwise_neighbor2 = src_masks.sum() * 0.
            loss_pairwise_neighbor3 = src_masks.sum() * 0.
            loss_pairwise_neighbor4 = src_masks.sum() * 0.

        # print('loss_proj term:', loss_prj_term)
        losses = {
            "loss_mask": loss_prj_term,
            "loss_bound": loss_pairwise,
            "loss_bound_neighbor": (loss_pairwise_neighbor + loss_pairwise_neighbor1 + loss_pairwise_neighbor2 + loss_pairwise_neighbor3 + loss_pairwise_neighbor4) * 0.1, # * 0.33
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks, images_lab_sim, images_lab_sim_nei, images_lab_sim_nei1, images_lab_sim_nei2, images_lab_sim_nei3, images_lab_sim_nei4):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks_proj,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        if loss == 'masks':
            return loss_map[loss](outputs, targets, indices, num_masks, images_lab_sim, images_lab_sim_nei, images_lab_sim_nei1, images_lab_sim_nei2, images_lab_sim_nei3, images_lab_sim_nei4)
        else:
            return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets, images_lab_sim, images_lab_sim_nei, images_lab_sim_nei1, images_lab_sim_nei2, images_lab_sim_nei3, images_lab_sim_nei4):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, images_lab_sim, images_lab_sim_nei, images_lab_sim_nei1, images_lab_sim_nei2, images_lab_sim_nei3, images_lab_sim_nei4))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, images_lab_sim, images_lab_sim_nei, images_lab_sim_nei1, images_lab_sim_nei2, images_lab_sim_nei3, images_lab_sim_nei4)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
