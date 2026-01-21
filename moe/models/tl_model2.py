from typing import Any
import lightning as L
import torch
import logging, os
from utils.wrapper import loss_wrapper, optim_wrapper, schedule_wrapper   
from utils.tools import cul_eer 
import numpy as np

class base_model(L.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.args = args
        self.model = model
        
        self.save_hyperparameters(self.args)
        
        self.model_optimizer = optim_wrapper.optimizer_wrap(self.args, self.model).get_optim()
        self.LRScheduler = schedule_wrapper.scheduler_wrap(self.model_optimizer, self.args).get_scheduler()
        
        self.args.model = model
        self.args.samloss_optim = self.model_optimizer
        self.loss_criterion, self.loss_optimizer, self.minimizor = loss_wrapper.loss_wrap(self.args).get_loss()
        
        self.logging_test = None
        self.logging_predict = None
        
    def forward(self, x,train=False):
        return self.model(x,train)
    
    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long, device=x.device)
        
        actual_batch_size = x.shape[0]
        
	    # 前向传播
#        if hasattr(self.model, 'delta_uq_enabled') and self.model.delta_uq_enabled:
#	        pred, _, moe_aux_loss, uncertainty = self.model(x, train=True, return_uncertainty=True)
#	        
#	        # 基础分类损失 - 确保是标量
#	        cls_loss = self.loss_criterion(pred, y)
#	        if cls_loss.dim() > 0 and cls_loss.numel() > 1:
#	            cls_loss = cls_loss.mean()
#	        
#	        # 不确定度正则化 - 确保是标量
#	        uncertainty_reg = torch.mean(uncertainty)
#	        if uncertainty_reg.dim() > 0 and uncertainty_reg.numel() > 1:
#	            uncertainty_reg = uncertainty_reg.mean()
#	        
#	        # 组合损失
#	        total_loss = cls_loss + self.model.args.uncertainty_weight * uncertainty_reg + self.args.loss_weight * moe_aux_loss
#	        
#	        # 记录指标
#	        self.log_dict({
#            "loss": total_loss,
#            "cls_loss": cls_loss,
#            "uncertainty": uncertainty_reg,
#            "moe_aux_loss": moe_aux_loss
#	        }, on_step=True, on_epoch=True, prog_bar=True, logger=True,
#           sync_dist=True, batch_size=actual_batch_size)
#        else:
	        # 如果不使用Delta-UQ，正常训练
        output = self.model(x, train=True)
        pred = output
        total_loss = self.loss_criterion(pred, y)
	        
        # 确保损失是标量
        if total_loss.dim() > 0 and total_loss.numel() > 1:
            total_loss = total_loss.mean()
	        
        self.log_dict({
	            "loss": total_loss
	        }, on_step=True, on_epoch=True, prog_bar=True, logger=True,
	           sync_dist=True, batch_size=actual_batch_size)
        
        return total_loss
    
#    def validation_step(self, batch, batch_idx):
#        x = batch[0]
#        y = batch[1]
#        filenames = batch[2]
#        
#        if not isinstance(y, torch.Tensor):
#            y = torch.tensor(y, dtype=torch.long, device=x.device)
#        
#        actual_batch_size = x.shape[0]
#        
#	    # 前向传播
#        if hasattr(self.model, 'delta_uq_enabled') and self.model.delta_uq_enabled:
#	        output = self.model(x, return_uncertainty=True)
#	        pred = output[0]
#	        uncertainty = output[3]  # 不确定度
#        else:
#	        output = self.model(x)
#	        pred = output[0]
#        
#        loss = self.loss_criterion(pred, y)
#        if loss.dim() > 0 and loss.numel() > 1:
#            loss = loss.mean()
#        
#        # 记录预测结果到验证日志
#        softmax_pred = torch.nn.functional.softmax(pred, dim=1)
#        with open(os.path.join(self.logger.log_dir, "dev.log"), 'a') as file:
#            for i in range(len(softmax_pred)):
#                file.write(f"{filenames[i]} {str(softmax_pred.cpu().numpy()[i][1])}\n")
#        
#        self.log("val_loss", loss, batch_size=actual_batch_size, sync_dist=True)
#        
#        return loss


    def validation_step(self, batch, batch_idx):
	    x = batch[0]
	    y = batch[1]
	    filenames = batch[2]
	    
	    if not isinstance(y, torch.Tensor):
	        y = torch.tensor(y, dtype=torch.long, device=x.device)
	    
	    actual_batch_size = x.shape[0]
	    
	    # 前向传播 - 验证时计算不确定性但不用于损失
#	    if hasattr(self.model, 'delta_uq_enabled') and self.model.delta_uq_enabled:
#	        output = self.model(x, return_uncertainty=True)
#	        pred = output[0]
#	        uncertainty = output[3]  # 不确定度
	        
	        # 关键修改：使用不确定性调整预测置信度
#	        if uncertainty is not None:
#	            # 将logits转换为概率
#	            softmax_pred = torch.nn.functional.softmax(pred, dim=1)
#	            
#	            # 使用不确定性调整概率：高不确定性 -> 降低置信度
#	            uncertainty_weight = 1.0 - uncertainty.unsqueeze(1) * 0.5  # 最大调整50%
#	            adjusted_probs = softmax_pred * uncertainty_weight
#	            
#	            # 重新归一化概率
#	            adjusted_probs = adjusted_probs / adjusted_probs.sum(dim=1, keepdim=True)
#	            
#	            # 记录调整后的概率
#	            final_probs = adjusted_probs
#	        else:
#	            final_probs = torch.nn.functional.softmax(pred, dim=1)
#	    else:
	    output = self.model(x,train=True)
#	    output = self.predict_with_uq(x)[0]	   
	    pred = output
	    final_probs = torch.nn.functional.softmax(pred, dim=1)
	    
	    # 计算损失 - 使用原始预测，不包含不确定性
	    loss = self.loss_criterion(pred, y)
	    if loss.dim() > 0 and loss.numel() > 1:
	        loss = loss.mean()
	    
	    # 记录调整后的预测结果到验证日志
	    with open(os.path.join(self.logger.log_dir, "dev.log"), 'a') as file:
	        for i in range(len(final_probs)):
	            file.write(f"{filenames[i]} {str(final_probs.cpu().numpy()[i][1])}\n")
	    
	    self.log("val_loss", loss, batch_size=actual_batch_size, sync_dist=True)
	    
	    return loss
    
#    def on_validation_epoch_end(self):
#        # 计算开发集EER
#        dev_eer = 0.
#        dev_tdcf = 0.
#        
#        dev_log_path = os.path.join(self.logger.log_dir, "dev.log")
#        if os.path.exists(dev_log_path):
#            with open(dev_log_path, 'r') as file:
#                lines = file.readlines()
#
#            if len(lines) > 0:
#                # 根据数据集名称选择相应的协议文件
#                dataset_name = getattr(self.args, 'dataset_name', '19LA')
#                if dataset_name == "19LA":
#                    label_file = "./data/ASVspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
#                    asv_file = "./data/ASVspoof2019_LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt"
#                elif dataset_name == "21LA":
#                    label_file = "./protocols/ASVspoof2021_LA/asvspoof2021_la.dev.trl.txt"
#
#                elif dataset_name == "21DF":
#                    label_file = "./protocols/ASVspoof2021_DF/asvspoof2021_df.dev.trl.txt"
#                else:  # ITW
#                    label_file = "./protocols/in_the_wild/in_the_wild.dev.trl.txt"
#                
#
#            
#            # 清空日志文件
#            with open(dev_log_path, 'w') as file:
#                pass
#        
#        self.log_dict({
#            "dev_eer": dev_eer,
#            "dev_tdcf": dev_tdcf,
#        }, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)


    
#    def test_step(self, batch, batch_idx):
#        x = batch[0]
#        y = batch[1]
#        filenames = batch[2]
#        
#        if not isinstance(y, torch.Tensor):
#            y = torch.tensor(y, dtype=torch.long, device=x.device)
#        
#	    # 前向传播
#        if hasattr(self.model, 'delta_uq_enabled') and self.model.delta_uq_enabled:
#	        output = self.model(x, return_uncertainty=True)
#	        pred = output[0]
#	        uncertainty = output[3]  # 不确定度
#        else:
#	        output = self.model(x)
#	        pred = output[0]
#        
#        loss = self.loss_criterion(pred, y)
#        if loss.dim() > 0 and loss.numel() > 1:
#            loss = loss.mean()
#        
#        softmax_pred = torch.nn.functional.softmax(pred, dim=1)
#        for i in range(len(filenames)):
#            self.logging_test.info(f"{filenames[i]} {str(softmax_pred.cpu().numpy()[i][1])}")
#        
#        self.log("test_loss", loss, batch_size=len(x), sync_dist=True)
#        
#        return {'loss': loss, 'y_pred': softmax_pred}


    def test_step(self, batch, batch_idx):
	    x, y, filenames = batch
	    output = self.model(x,train=True)
	    pred = output
	
	    softmax_pred = torch.nn.functional.softmax(pred, dim=1)
	
#	    if uncertainty is not None:
#	        # 确保 uncertainty 为 [batch, 1]
#	        if uncertainty.dim() == 1:
#	            uncertainty = uncertainty.unsqueeze(1)
#	        # 按样本置信度衰减
#	        uncertainty_weight = 1.0 - 0.5 * uncertainty
#	        adjusted_probs = softmax_pred * uncertainty_weight
#	        final_probs = adjusted_probs / adjusted_probs.sum(dim=1, keepdim=True)
#	    else:
	    final_probs = softmax_pred
	
	    # 记录bonafide分数
	    bonafide_scores = final_probs[:, 1]
	    for i in range(len(filenames)):
	        self.logging_test.info(f"{filenames[i]} {bonafide_scores[i].item()}")
	
	    return {'y_pred': final_probs}


    
    
    def predict_step(self, batch, batch_idx):
        x = batch[0]
        filenames = batch[2]
        
	    # 前向传播
#        if hasattr(self.model, 'delta_uq_enabled') and self.model.delta_uq_enabled:
#	        output = self.model(x, return_uncertainty=True)
#	        pred = output[0]
#	        uncertainty = output[3]  # 不确定度
#        else:
        output = self.model(x)
        pred = output
        
        softmax_pred = torch.nn.functional.softmax(pred, dim=1)
        for i in range(len(filenames)):
            self.logging_predict.info(f"{filenames[i]} {str(softmax_pred.cpu().numpy()[i][1])}")
        
        return
    
    def on_test_start(self):
        dataset_name = getattr(self.args, 'dataset_name', '19LA')
        self.logging_test = logging.getLogger(f"logging_test_{dataset_name}")
        self.logging_test.setLevel(logging.INFO)
        hdl = logging.FileHandler(os.path.join(self.logger.log_dir, f"infer_{dataset_name}.log"))
        hdl.setFormatter(logging.Formatter('%(message)s'))
        self.logging_test.addHandler(hdl)
    
    def on_predict_start(self):
        testset = getattr(self.args, 'testset', 'unknown')
        self.logging_predict = logging.getLogger(f"logging_predict_{testset}")
        self.logging_predict.setLevel(logging.INFO)
        hdlx = logging.FileHandler(os.path.join(self.logger.log_dir, f"infer_{testset}.log"))
        hdlx.setFormatter(logging.Formatter('%(message)s'))
        self.logging_predict.addHandler(hdlx)
    
    def configure_optimizers(self):
        if self.LRScheduler is not None:
            return {
                "optimizer": self.model_optimizer,
                'lr_scheduler': self.LRScheduler, 
                'monitor': 'dev_eer'
            }
        else:
            return {
                "optimizer": self.model_optimizer,
            }