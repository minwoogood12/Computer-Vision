def topk_mask(self, images_lab_sim, k):
        images_lab_sim_mask = torch.zeros_like(images_lab_sim)
        topk, indices = torch.topk(images_lab_sim, k, dim =1) # 1, 3, 5, 7
        #n배치 개수만큼 반복해서 모든 배치에서 top-k를 선정해서 나머지는 0으로 만듬
        images_lab_sim_mask = images_lab_sim_mask.scatter(1, indices, topk)
        return images_lab_sim_mask

def loss_masks_proj(self, outputs, targets, num_masks,
                    images_lab_sim, 
                    images_lab_sim_nei, 
                    images_lab_sim_nei1,
                    images_lab_sim_nei2, 
                    images_lab_sim_nei3,
                    images_lab_sim_nei4,
                   images_lab_sim_nei5,
                   images_lab_sim_nei6,
                   images_lab_sim_nei7):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        
        
        self._iter += 1

        '''
        indices = [
          (tensor([0, 1]), tensor([2, 0])),
          (tensor([3]), tensor([1]))
        ] 총 배치 개수만큼 리스트로

        '''
        '''
        batch_idx = tensor([0, 0, 1])  배치 번호
        src_idx   = tensor([0, 2, 1]) 쿼리 번호
        '''
        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"] 
        #[B,Q,H,W]
        src_masks = src_masks[src_idx]
        #[N,T,H,W]
        # Modified to handle video
        target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)]).to(src_masks)
        #[N,T,H,W]

  
        images_lab_sim = torch.cat(images_lab_sim, dim =0)
        #List[Tensor(1, K^2,H,W) * (B*T)]를 1차원에 길이만큼 이어붙임 => Tensor(B*T,K^2,H,W)
        images_lab_sim_nei = torch.cat(images_lab_sim_nei, dim=0)
        #Tensor(B,K^2,H,W)
        images_lab_sim_nei1 = torch.cat(images_lab_sim_nei1, dim=0)
        images_lab_sim_nei2 = torch.cat(images_lab_sim_nei2, dim=0)
        images_lab_sim_nei3 = torch.cat(images_lab_sim_nei3, dim=0)
        images_lab_sim_nei4 = torch.cat(images_lab_sim_nei4, dim=0)
        images_lab_sim_nei5 = torch.cat(images_lab_sim_nei5, dim=0)
        images_lab_sim_nei6 = torch.cat(images_lab_sim_nei6, dim=0)
        images_lab_sim_nei7 = torch.cat(images_lab_sim_nei7, dim=0)

        images_lab_sim = images_lab_sim.view(-1, target_masks.shape[1], images_lab_sim.shape[-3], images_lab_sim.shape[-2], images_lab_sim.shape[-1])
        #Tensor(B,T,K^2,H,W)
        images_lab_sim_nei = images_lab_sim_nei.unsqueeze(1)
        #Tensor(B,1,K^2,H,W)
        images_lab_sim_nei1 = images_lab_sim_nei1.unsqueeze(1)
        images_lab_sim_nei2 = images_lab_sim_nei2.unsqueeze(1)
        images_lab_sim_nei3 = images_lab_sim_nei3.unsqueeze(1)
        images_lab_sim_nei4 = images_lab_sim_nei4.unsqueeze(1)
        images_lab_sim_nei5 = images_lab_sim_nei5.unsqueeze(1)
        images_lab_sim_nei6 = images_lab_sim_nei6.unsqueeze(1)
        images_lab_sim_nei7 = images_lab_sim_nei7.unsqueeze(1)

        images_lab_sim = torch.cat([images_lab_sim[ind][None] for ind in src_idx[0].tolist()]).flatten(0, 1)
        #[N*T,K^2,H,W]
        images_lab_sim_nei = self.topk_mask(images_lab_sim_nei.flatten(0, 1), 5)
        #Tensor(B,1,K^2,H,W) 배치 개수 중에 쌍이 된 배치수만큼 cat하고 각각 배치에 대해서 k^2중에 5개만 고르고 나머지 0으로 만들고 리턴
        #[N,K^2,H,W]
        images_lab_sim_nei1 = self.topk_mask(images_lab_sim_nei1.flatten(0, 1), 5)
        images_lab_sim_nei2 = self.topk_mask(images_lab_sim_nei2.flatten(0, 1), 5)
        images_lab_sim_nei3 = self.topk_mask(images_lab_sim_nei3.flatten(0, 1), 5)
        images_lab_sim_nei4 = self.topk_mask(images_lab_sim_nei4.flatten(0, 1), 5)
        images_lab_sim_nei5 = self.topk_mask(images_lab_sim_nei5.flatten(0, 1), 5)
        images_lab_sim_nei6 = self.topk_mask(images_lab_sim_nei6.flatten(0, 1), 5)
        images_lab_sim_nei7 = self.topk_mask(images_lab_sim_nei7.flatten(0, 1), 5)
  
      
        
        k_size = 3 

        if src_masks.shape[0] > 0:
            pairwise_losses_neighbor = compute_pairwise_term_neighbor(
                src_masks[:,:1], src_masks[:,1:2], k_size, 3
            ) 
            #예측마스크[N, T, H, W] 의 객체 별로 각가 0번프레임[N,1,H,W], 1번프레임[N,1,H,W] 비교
            #[N, H, W] 예측된 마스크 n개에 대해 0번 1번 프레임이 얼마나 비슷한 예측인지 비교한 map (0~0.69) 0이면 일치, 0.69면 일치안함
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
                src_masks[:,4:5], src_masks[:,5:6], k_size, 3
            )
            pairwise_losses_neighbor5 = compute_pairwise_term_neighbor(
                src_masks[:,5:6], src_masks[:,6:7], k_size, 3
            )
            pairwise_losses_neighbor6 = compute_pairwise_term_neighbor(
                src_masks[:,6:7], src_masks[:,7:8], k_size, 3
            )
            pairwise_losses_neighbor7 = compute_pairwise_term_neighbor(
                src_masks[:,7:8], src_masks[:,0:1], k_size, 3
            )
            
        # print('pairwise_losses_neighbor:', pairwise_losses_neighbor.shape)
        src_masks = src_masks.flatten(0, 1)[:, None]
        #0,1차원을 합치고 1차원 추가 생성 즉 [N*T, 1, H, W]
        target_masks = target_masks.flatten(0, 1)[:, None]
        #0,1차원을 합치고 1차원 추가 생성 즉 [N*T, 1, H, W]
        target_masks = F.interpolate(target_masks, (src_masks.shape[-2], src_masks.shape[-1]), mode='bilinear')
        #정답마스크해상도를 예측에 맞게 리사이즈
        # images_lab_sim = F.interpolate(images_lab_sim, (src_masks.shape[-2], src_masks.shape[-1]), mode='bilinear')
        
        
        if src_masks.shape[0] > 0:
            loss_prj_term = compute_project_term(src_masks.sigmoid(), target_masks)  
            #공간로스방식 : 각각 x,y로 projection하고 로스의 평균 반환 스칼라값.
            pairwise_losses = compute_pairwise_term(
                src_masks, k_size, 2
            )
            #한 프레임 내에서 주변픽셀과의 유사도 구하기 
            #[N*T, H, W]

            weights = (images_lab_sim >= 0.3).float() * target_masks.float()
            #프레임 내 유사도가 0.3이상이고 target마스크내에서 가중치계산
            #image_lab_sim : [N*T,K^2,H,W]
            #target_masks : [N*T,1,H,W]
            #결과 [N*T, K², H, W]

            target_masks_sum = target_masks.reshape(pairwise_losses_neighbor.shape[0], 8, target_masks.shape[-2], target_masks.shape[-1]).sum(dim=1, keepdim=True)
            #target_masks 를 다시 [N*T,1,H,W] -> [N,T,H,W]로 바꿈 -> T축으로 합
            # => [N,1,H,W] #8 : 8개의 프레임의미

            target_masks_sum = (target_masks_sum >= 1.0).float() # ori is 1.0
            #[N,1,H,W]에서 한번이라도 프레임에 등장했나 등장하면 1아니면 0
            #[N,1,H,W]

            weights_neighbor = (images_lab_sim_nei >= 0.05).float() * target_masks_sum # ori is 0.5, 0.01, 0.001, 0.005, 0.0001, 0.02, 0.05, 0.075, 0.1 , dy 0.5
            #images_lab_sim_nei : [N,K^2,H,W] 각 값이 0.05이상이면 1 아니면 0으로 변환
            #[N,K^2,H,W]

            weights_neighbor1 = (images_lab_sim_nei1 >= 0.05).float() * target_masks_sum # ori is 0.5, 0.01, 0.001, 0.005, 0.0001, 0.02, 0.05, 0.075, 0.1, dy 0.5
            weights_neighbor2 = (images_lab_sim_nei2 >= 0.05).float() * target_masks_sum # ori is 0.5, 0.01, 0.001, 0.005, 0.0001, 0.02, 0.05, 0.075, 0.1, dy 0.5
            weights_neighbor3 = (images_lab_sim_nei3 >= 0.05).float() * target_masks_sum
            weights_neighbor4 = (images_lab_sim_nei4 >= 0.05).float() * target_masks_sum
            weights_neighbor5 = (images_lab_sim_nei5 >= 0.05).float() * target_masks_sum
            weights_neighbor6 = (images_lab_sim_nei6 >= 0.05).float() * target_masks_sum
            weights_neighbor7 = (images_lab_sim_nei7 >= 0.05).float() * target_masks_sum

            warmup_factor = min(self._iter.item() / float(self._warmup_iters), 1.0) #1.0
            #학습 수 설정 나중 갈수록 더 가중치있게

            loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)
            #한 프레임내의 주변픽셀과의 유사도[N*T, H, W] * [N*T, K², H, W] / 전체에서 반영된 픽셀의 총합
            loss_pairwise_neighbor = (pairwise_losses_neighbor * weights_neighbor).sum() / weights_neighbor.sum().clamp(min=1.0) * warmup_factor
            #(예측마스크 0번과 1번마스크 비교 * 0번과 1번마스크사이의 sim_nei 비교 가중치값)의 합
            loss_pairwise_neighbor1 = (pairwise_losses_neighbor1 * weights_neighbor1).sum() / weights_neighbor1.sum().clamp(min=1.0) * warmup_factor
            loss_pairwise_neighbor2 = (pairwise_losses_neighbor2 * weights_neighbor2).sum() / weights_neighbor2.sum().clamp(min=1.0) * warmup_factor
            loss_pairwise_neighbor3 = (pairwise_losses_neighbor3 * weights_neighbor3).sum() / weights_neighbor3.sum().clamp(min=1.0) * warmup_factor
            loss_pairwise_neighbor4 = (pairwise_losses_neighbor4 * weights_neighbor4).sum() / weights_neighbor4.sum().clamp(min=1.0) * warmup_factor
            loss_pairwise_neighbor5 = (pairwise_losses_neighbor5 * weights_neighbor5).sum() / weights_neighbor5.sum().clamp(min=1.0) * warmup_factor
            loss_pairwise_neighbor6 = (pairwise_losses_neighbor6 * weights_neighbor6).sum() / weights_neighbor6.sum().clamp(min=1.0) * warmup_factor
            loss_pairwise_neighbor7 = (pairwise_losses_neighbor7 * weights_neighbor7).sum() / weights_neighbor7.sum().clamp(min=1.0) * warmup_factor

                           
        else:
            loss_prj_term = src_masks.sum() * 0.
            loss_pairwise = src_masks.sum() * 0.
            loss_pairwise_neighbor = src_masks.sum() * 0.
            loss_pairwise_neighbor1 = src_masks.sum() * 0.
            loss_pairwise_neighbor2 = src_masks.sum() * 0.
            loss_pairwise_neighbor3 = src_masks.sum() * 0.
            loss_pairwise_neighbor4 = src_masks.sum() * 0.
            loss_pairwise_neighbor5 = src_masks.sum() * 0.
            loss_pairwise_neighbor6 = src_masks.sum() * 0.
            loss_pairwise_neighbor7 = src_masks.sum() * 0.
                

        # print('loss_proj term:', loss_prj_term)
        losses = {
            "loss_mask": loss_prj_term,
            "loss_bound": loss_pairwise,
            "loss_bound_neighbor": (loss_pairwise_neighbor
                                    + loss_pairwise_neighbor1 
                                    + loss_pairwise_neighbor2 
                                    + loss_pairwise_neighbor3
                                    + loss_pairwise_neighbor4
                                    + loss_pairwise_neighbor5
                                    + loss_pairwise_neighbor6
                                    + loss_pairwise_neighbor7
                                   ) * 0.1, # * 0.33
        }

        del src_masks
        del target_masks
        return losses
