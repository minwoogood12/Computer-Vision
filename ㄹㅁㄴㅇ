 <<SAM2>>
<train.py>
1. if __name__ : config 등록
2. main : cluster 0(로컬) or 1 선택 
3. single_proc_runner() :  GPU 1, 여러개 구분
4. single_proc_run() : trainer객체 생성 및 trainer.run.() 시작

<trainer.py>
1. run() : train, train_only, val 3가지로 분류 (train_only : run_train())
2. run_train() : 
while self.epoch < self.max_epochs:현재 epoch기준 Max_epoch만큼 반복
  dataloader = Iterable(커스텀된 리스트 느낌) = [
    {'video_images': Tensor [B, T, 3, H, W],       # 영상 프레임 시퀀스
    'gt_masks': Tensor [B, T, N, H, W],           # ground truth segmentation masks
    'gt_boxes': Tensor [B, T, N, 4],              # GT boxes (optional, not always used)
    'prompt_points': Tensor [B, T, N, num_pts, 2],# point prompt 위치 num_pts : 객체의 여러 포인트 , 2 : 좌표 2곳
    'prompt_labels': Tensor [B, T, N, num_pts],   # 각 point의 foreground(1)/background(0) 레이블
    'frame_ids': Tensor [B, T],                   # 프레임 인덱스
    'meta': {...},                                # 시퀀스 이름 등 메타 정보
    ...}
   ... 
  ] #length : Batch 개수
  outs = self.train_epoch(dataloader) -> 최종 log결과 딕셔너리
이후 로그저장 및, 체크포인터 저장, epoch +1 정도만 실행

3. train_epoch(self, train_loader) : 
모니터링 객체 초기화, phase, loss meter 초기화
for data_iter, batch in enumerate(train_loader) : 전체 배치 수만큼 반복 
  self.run_step(batch, phase, loss_mts, extra_loss_mts) : phase(현재단계 train, val...), loss_mts(loss로그 기록용), extra_loss_mts(부가적 loss 기록용)
  self.scaler.step(self.optim.optimizer) backward에서 계산된 gradient 파라미터 적용
  self.scaler.update() 다음 step을위한 scaler 업데이트
[최종 로그값 딕셔너리 리턴] 
=> 매 배치마다 gradient파라미터 업데이트 (_run_step에서 매 배치마다 backward()) => 즉 배치한번마다 backward, update(step) 실행

4. _run_step
loss_dict, batch_size, extra_losses = self._step(batch, self.model, phase,) 이떄 loss_dict : 1개의 값(주요 1개 로스) , extra losses(나머지 로스들)
self.scaler.scale(loss).backwardㅇㅇ
