<train_net_video.py>
1. __name__ == "__main__":
launch(main,
       args.num_gpus,
       num_machines=args.num_machines
       ....) #main 호출

2. def main(args):
cfg = setup(args) #학습에 필요한 config 객체 생성
trainer = Trainer(cfg) #model, dataloader, optimizer 등 설정
trainer.resume_or_load(resume=args.resume) #checkpoints 설정
return trainer.train() #실제 학습 시작
