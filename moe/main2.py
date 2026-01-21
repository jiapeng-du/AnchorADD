import os
import yaml
import shutil
import torch
from utils.arg_parse import f_args_parsed, set_random_seed
from datetime import datetime
import lightning as L
import importlib
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch import loggers as pl_loggers

def main():
    # arguments initialization
    args = f_args_parsed()

    # 新增：数据集名称参数
    if not hasattr(args, 'dataset_name'):
        args.dataset_name = "DF21"  # 默认使用19LA
    
    # config gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

    # random seed initialization and gpu seed
    set_random_seed(args.seed, args)

    # config the base model containing train eval test and inference funtion
    tl_model = importlib.import_module(args.tl_model)

    # config the data module containing the train set, dev set and test set
    dm_module = importlib.import_module(args.data_module)
    asvspoof_dm = dm_module.asvspoof_dataModule(args=args)

    if True:
        # ⭐train
        if not args.inference:
            # import model.py
            prj_model = importlib.import_module(args.module_model)

            # model
            model = prj_model.Model(args)

            # init model, including loss func and optim
            customed_model_wrapper = tl_model.base_model(
                model=model,
                args=args
            )

            # config logdir - 在日志目录中包含数据集名称
            log_dir = f"{args.savedir}/{args.dataset_name}_uq"
            tb_logger = pl_loggers.TensorBoardLogger(log_dir, name="")
            
            # model initialization
            trainer = L.Trainer(
                max_epochs=args.epochs,
                accelerator="gpu",
                devices=1,
                strategy="auto",
                log_every_n_steps=1,
                callbacks=[
	                # dev损失无下降就提前停止
	                EarlyStopping('val_loss',patience=args.no_best_epochs,mode="min",verbose=True,log_rank_zero_only=True),
	                # 模型按照最低val_loss来保存
	                ModelCheckpoint(monitor='val_loss',
	                                save_top_k=1,
	                                save_weights_only=True,mode="min",filename='best_model-{epoch:02d}-{dev_eer:.4f}-{val_loss:.4f}'),
	                LearningRateMonitor(logging_interval='epoch',log_weight_decay=True),
	                ],
                check_val_every_n_epoch=1,
                logger=tb_logger,
                enable_progress_bar=False
            )
            
            trainer.fit(
                model=customed_model_wrapper, 
                datamodule=asvspoof_dm
            )
            
            # 在当前数据集上测试
#            trainer.test(
#                model=customed_model_wrapper,
#                datamodule=asvspoof_dm
#            )
        else:
            # 推理模式
            checkpointpath = args.trained_model
            args.savedir = checkpointpath

            # gain model
            ymlconf = os.path.join(checkpointpath, "hparams.yaml")
            with open(ymlconf, "r") as f_yaml:
                parser1 = yaml.safe_load(f_yaml)
            
            infer_m = importlib.import_module(parser1["module_model"])
            test_dm_module = importlib.import_module(parser1["data_module"])
            
            infer_model = infer_m.Model(args)

            print(parser1)

            # 加载检查点
            ckpt_files = [file for file in os.listdir(checkpointpath + "/checkpoints/") if file.endswith(".ckpt")]
            customed_model = tl_model.base_model.load_from_checkpoint(
                checkpoint_path=os.path.join(f"{checkpointpath}/checkpoints/", ckpt_files[0]),
                model=infer_model,
                args=args,
                strict=False
            )
            
            inferer = L.Trainer(logger=pl_loggers.TensorBoardLogger(args.savedir, name=""))

    
            test_datasets = args.dataset_name
            print(f"\n=== Testing on {args.dataset_name} ===")
#            args.dataset_name = dataset
            test_asvspoof_dm = test_dm_module.asvspoof_dataModule(args=args)
           
           # 测试当前数据集
            inferer.test(
               model=customed_model,
               datamodule=test_asvspoof_dm
            )

            # 预测模式（保持原有逻辑）
#            predict_datasets = {
#                "LA21": "ASVspoof2021_LA_eval/ASVspoof2021.LA.cm.eval.trl.txt",
#                "DF21": "ASVspoof2021_DF_eval/ASVspoof2021.DF.cm.eval.trl.txt", 
#                "ITW": "release_in_the_wild/label.txt"
#            }
#            
#            for predict_name, protocol_file in predict_datasets.items():
#                if predict_name == args.dataset_name:
#                    print(f"\n=== Predicting on {predict_name} ===")
#                    customed_model.args.testset = predict_name
#                    test_asvspoof_dm = test_dm_module.asvspoof_dataModule(args=args)
#                    inferer.predict(
#                    	model=customed_model,
#                    	datamodule=test_asvspoof_dm
#                    )

            # 清理和重命名日志文件
            current_time = datetime.now()
            time_str = current_time.strftime("%Y_%m_%d_%H_%M_%S")
            inferfolder = os.path.join(checkpointpath, f"infer_{time_str}")
            if not os.path.exists(inferfolder):
                os.makedirs(inferfolder)
            folder_a = os.path.join(checkpointpath, "version_0")
            for filename in os.listdir(folder_a):
                if filename.endswith('.log'):
                    original_path = os.path.join(folder_a, filename)
                    destination_path = os.path.join(inferfolder, filename)
                    shutil.move(original_path, destination_path)
            shutil.rmtree(folder_a)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()