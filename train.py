import os
import time
import torch

from utils import Train_Report, TestReport, SaveManager


class Trainer:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        if self.config.save_train_img:
            self.save_manager = SaveManager(config)
        self.criterion = torch.nn.L1Loss()

        milestones = [260, 360, 380, 390]
        # optimizer and scheduler for degradation learning network
        self.optimizer_D = torch.optim.Adam(self.model.degradation_learning_network.parameters(), lr=self.config.lr)
        self.scheduler_D = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_D, milestones=milestones, gamma=0.5, last_epoch=-1)

        # optimizer and scheduler for restoration network
        if self.config.stage == 2:
            self.optimizer_R = torch.optim.Adam(self.model.restoration_network.parameters(), lr=self.config.lr)
            self.scheduler_R = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_R, milestones=milestones, gamma=0.5, last_epoch=-1)

        self.checkpoint_path = os.path.join(self.config.save_dir, f'model_stage{self.config.stage}')
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        self.model.cuda()

    def save_checkpoint(self, epoch):
        D_state_dict = {'epoch': epoch,
                        'model_D_state_dict': self.model.degradation_learning_network.state_dict(),
                        'optimizer_D_state_dict': self.optimizer_D.state_dict(),
                        'scheduler_D_state_dict': self.scheduler_D.state_dict()}
        torch.save(D_state_dict, self.checkpoint_path + '/model_D_latest.pt')
        torch.save(D_state_dict, self.checkpoint_path + '/model_D_' + str(epoch) + '.pt')

        if self.config.stage == 2:
            R_state_dict = {'epoch': epoch,
                            'model_R_state_dict': self.model.restoration_network.state_dict(),
                            'optimizer_R_state_dict': self.optimizer_R.state_dict(),
                            'scheduler_R_state_dict': self.scheduler_R.state_dict()}
            torch.save(R_state_dict, self.checkpoint_path + '/model_R_latest.pt')
            torch.save(R_state_dict, self.checkpoint_path + '/model_R_' + str(epoch) + '.pt')

    def save_best_model(self, epoch):
        D_state_dict = {'epoch': epoch,
                        'model_D_state_dict': self.model.degradation_learning_network.state_dict(),
                        'optimizer_D_state_dict': self.optimizer_D.state_dict(),
                        'scheduler_D_state_dict': self.scheduler_D.state_dict()}
        torch.save(D_state_dict, self.checkpoint_path + '/model_D_best.pt')

        if self.config.stage == 2:
            R_state_dict = {'epoch': epoch,
                            'model_R_state_dict': self.model.restoration_network.state_dict(),
                            'optimizer_R_state_dict': self.optimizer_R.state_dict(),
                            'scheduler_R_state_dict': self.scheduler_R.state_dict()}
            torch.save(R_state_dict, self.checkpoint_path + '/model_R_best.pt')

    def load_checkpoint(self, epoch=None):
        if epoch is None:
            D_state_dict = torch.load(self.checkpoint_path + '/model_D_latest.pt')
            self.model.degradation_learning_network.load_state_dict(D_state_dict['model_D_state_dict'])
            self.optimizer_D.load_state_dict(D_state_dict['optimizer_D_state_dict'])
            self.scheduler_D.load_state_dict(D_state_dict['scheduler_D_state_dict'])
            last_epoch = D_state_dict['epoch']
            print(f'load degradation learning network status from {self.checkpoint_path}/model_D_latest.pt, epoch: {last_epoch}')

            if self.config.stage == 2:
                R_state_dict = torch.load(self.checkpoint_path + '/model_R_latest.pt')
                self.model.restoration_network.load_state_dict(R_state_dict['model_R_state_dict'])
                self.optimizer_R.load_state_dict(R_state_dict['optimizer_R_state_dict'])
                self.scheduler_R.load_state_dict(R_state_dict['scheduler_R_state_dict'])
                last_epoch = R_state_dict['epoch']
                print(f'load restoration network status from {self.checkpoint_path}/model_R_latest.pt, epoch: {last_epoch}')

        else:
            D_state_dict = torch.load(self.checkpoint_path + '/model_D_' + str(epoch) + '.pt')
            self.model.degradation_learning_network.load_state_dict(D_state_dict['model_D_state_dict'])
            self.optimizer_D.load_state_dict(D_state_dict['optimizer_D_state_dict'])
            self.scheduler_D.load_state_dict(D_state_dict['scheduler_D_state_dict'])
            last_epoch = D_state_dict['epoch']
            print(f'load degradation learning network status from {self.checkpoint_path}/model_D_{epoch}.pt, epoch: {last_epoch}')

            if self.config.stage == 2:
                R_state_dict = torch.load(self.checkpoint_path + '/model_R_' + str(epoch) + '.pt')
                self.model.restoration_network.load_state_dict(R_state_dict['model_R_state_dict'])
                self.optimizer_R.load_state_dict(R_state_dict['optimizer_R_state_dict'])
                self.scheduler_R.load_state_dict(R_state_dict['scheduler_R_state_dict'])
                last_epoch = R_state_dict['epoch']
                print(f'load restoration network status from {self.checkpoint_path}/model_R_{epoch}.pt, epoch: {last_epoch}')

        return last_epoch

    def load_best_model(self):
        D_state_dict = torch.load(self.checkpoint_path + '/model_D_best.pt')
        self.model.degradation_learning_network.load_state_dict(D_state_dict['model_D_state_dict'])
        print(f'load degradation learning network status from {self.checkpoint_path}/model_D_best.pt, epoch: {D_state_dict["epoch"]}')

        if self.config.stage == 2:
            R_state_dict = torch.load(self.checkpoint_path + '/model_R_best.pt')
            self.model.restoration_network.load_state_dict(R_state_dict['model_R_state_dict'])
            print(f'load restoration network status from {self.checkpoint_path}/model_R_best.pt, epoch: {R_state_dict["epoch"]}')

    def load_best_stage1_model(self):
        path = self.checkpoint_path.replace(f'model_stage{self.config.stage}', 'model_stage1')
        state_dict = torch.load(path + '/model_D_best.pt')
        self.model.degradation_learning_network.load_state_dict(state_dict['model_D_state_dict'])
        self.optimizer_D.load_state_dict(state_dict['optimizer_D_state_dict'])
        self.scheduler_D.load_state_dict(state_dict['scheduler_D_state_dict'])
        print(f'load degradation learning network status  from {path}/model_D_best.pt, epoch: {state_dict["epoch"]}')

    def train(self, dataloader, train_log, global_step):
        self.model.train()
        report = Train_Report()
        start = time.time()

        for idx, (lr_blur_seq, hr_sharp_seq, lr_sharp_seq, flow) in enumerate(dataloader):
            lr_blur_seq = lr_blur_seq.cuda()
            hr_sharp_seq = hr_sharp_seq.cuda()
            lr_sharp_seq = lr_sharp_seq.cuda()
            flow = flow.cuda()

            result_dict = self.model(lr_blur_seq, hr_sharp_seq)

            batch_size, _, t, _, _ = lr_blur_seq.shape

            # pretrain degradation learning network
            if self.config.stage == 1:
                recon_loss = self.criterion(result_dict['recon'], lr_blur_seq[:, :, t//2, :, :])
                hr_warping_loss = self.config.hr_warping_loss_weight * self.criterion(result_dict['hr_warp'], hr_sharp_seq[:, :, t//2:t//2+1, :, :].repeat([1,1,t,1,1]))
                # RAFT pseudo-GT optical flow loss
                flow_loss = self.config.flow_loss_weight * self.criterion(result_dict['image_flow'], flow)
                # TA loss for degradation learning network
                D_TA_loss = self.config.D_TA_loss_weight * self.criterion(result_dict['F_sharp_D'], lr_sharp_seq)

                total_loss = recon_loss + hr_warping_loss + flow_loss + D_TA_loss

                self.optimizer_D.zero_grad()
                total_loss.backward()
                self.optimizer_D.step()

                report.update(batch_size, 0, recon_loss.item(), hr_warping_loss.item(), 0, flow_loss.item(), D_TA_loss.item(), 0, total_loss.item())

            # train full network
            elif self.config.stage == 2:
                restoration_loss = self.criterion(result_dict['output'], hr_sharp_seq[:, :, t//2, :, :])
                recon_loss = self.config.Net_D_weight * self.criterion(result_dict['recon'], lr_blur_seq[:, :, t//2, :, :])
                lr_warping_loss = self.config.lr_warping_loss_weight * self.criterion(result_dict['lr_warp'], lr_blur_seq[:, :, t//2:t//2 + 1, :, :].repeat([1,1,t,1,1]))
                hr_warping_loss = self.config.Net_D_weight * self.config.hr_warping_loss_weight * self.criterion(result_dict['hr_warp'], hr_sharp_seq[:, :, t//2:t//2+1, :, :].repeat([1,1,t,1,1]))
                # RAFT pseudo-GT optical flow loss
                flow_loss = self.config.Net_D_weight * self.config.flow_loss_weight * self.criterion(result_dict['image_flow'], flow)
                # TA loss for degradation learning network and restoration network
                R_TA_loss = self.config.R_TA_loss_weight * self.criterion(result_dict['F_sharp_R'], lr_sharp_seq)
                D_TA_loss = self.config.Net_D_weight * self.config.D_TA_loss_weight * self.criterion(result_dict['F_sharp_D'], lr_sharp_seq)
                
                total_loss = restoration_loss + recon_loss + hr_warping_loss + lr_warping_loss + flow_loss + R_TA_loss + D_TA_loss

                self.optimizer_D.zero_grad()
                self.optimizer_R.zero_grad()
                total_loss.backward()
                self.optimizer_D.step()
                self.optimizer_R.step()

                report.update(batch_size, restoration_loss.item(), recon_loss.item(), hr_warping_loss.item(), lr_warping_loss.item(), flow_loss.item(), D_TA_loss.item(), R_TA_loss.item(), total_loss.item())

            global_step += 1

            if global_step % 100 == 0 or idx == len(dataloader) - 1:
                lr_D = self.scheduler_D.optimizer.state_dict()['param_groups'][0]['lr']
                lr_R = self.scheduler_R.optimizer.state_dict()['param_groups'][0]['lr'] if self.config.stage == 2 else None

                period_time = time.time() - start
                prefix_str = f'[{global_step}/{len(dataloader) * self.config.num_epochs}]\t'
                result_str = report.result_str(lr_D, lr_R, period_time)

                train_log.write(prefix_str + result_str)
                start = time.time()
                report.__init__()

                if self.config.save_train_img:
                    if self.config.stage == 1:
                        src = [lr_blur_seq[:, :, t // 2, :, :], result_dict['recon']]
                    elif self.config.stage == 2:
                        src = [lr_blur_seq[:, :, t // 2, :, :], result_dict['recon'], result_dict['output'], hr_sharp_seq[:, :, t // 2, :, :]]
                    self.save_manager.save_batch_images(src, batch_size, global_step)

        self.scheduler_D.step()
        if self.config.stage == 2:
            self.scheduler_D.step()

        return global_step

    def validate(self, dataloader, val_log, epoch):
        self.model.eval()
        report = Train_Report()
        start = time.time()

        with torch.no_grad():
            for idx, (lr_blur_seq, hr_sharp_seq, lr_sharp_seq, flow) in enumerate(dataloader):
                lr_blur_seq = lr_blur_seq.cuda()
                hr_sharp_seq = hr_sharp_seq.cuda()
                lr_sharp_seq = lr_sharp_seq.cuda()
                flow = flow.cuda()

                result_dict = self.model(lr_blur_seq, hr_sharp_seq)

                batch_size, _, t, _, _ = lr_blur_seq.shape

                if self.config.stage == 1:
                    recon_loss = self.criterion(result_dict['recon'], lr_blur_seq[:, :, t // 2, :, :])
                    hr_warping_loss = self.config.hr_warping_loss_weight * self.criterion(result_dict['hr_warp'], hr_sharp_seq[:, :, t // 2:t // 2 + 1, :, :].repeat([1, 1, t, 1, 1]))
                    flow_loss = self.config.flow_loss_weight * self.criterion(result_dict['image_flow'], flow)
                    D_TA_loss = self.config.D_TA_loss_weight * self.criterion(result_dict['F_sharp_D'], lr_sharp_seq)

                    total_loss = recon_loss + hr_warping_loss + flow_loss + D_TA_loss
                    report.update(batch_size, 0, recon_loss.item(), hr_warping_loss.item(), 0, flow_loss.item(), D_TA_loss.item(), 0, total_loss.item())
                    report.update_recon_metric(result_dict['recon'], lr_blur_seq[:, :, t // 2, :, :])

                elif self.config.stage == 2:
                    restoration_loss = self.criterion(result_dict['output'], hr_sharp_seq[:, :, t // 2, :, :])
                    recon_loss = self.config.Net_D_weight * self.criterion(result_dict['recon'], lr_blur_seq[:, :, t // 2, :, :])
                    lr_warping_loss = self.config.lr_warping_loss_weight * self.criterion(result_dict['lr_warp'], lr_blur_seq[:, :, t // 2:t // 2 + 1, :, :].repeat([1, 1, t, 1, 1]))
                    hr_warping_loss = self.config.Net_D_weight * self.config.hr_warping_loss_weight * self.criterion(result_dict['hr_warp'], hr_sharp_seq[:, :, t // 2:t // 2 + 1, :, :].repeat([1, 1, t, 1, 1]))
                    flow_loss = self.config.Net_D_weight * self.config.flow_loss_weight * self.criterion(result_dict['image_flow'], flow)
                    R_TA_loss = self.config.R_TA_loss_weight * self.criterion(result_dict['F_sharp_R'], lr_sharp_seq)
                    D_TA_loss = self.config.Net_D_weight * self.config.D_TA_loss_weight * self.criterion(result_dict['F_sharp_D'], lr_sharp_seq)

                    total_loss = restoration_loss + recon_loss + hr_warping_loss + lr_warping_loss + flow_loss + R_TA_loss + D_TA_loss
                    report.update(batch_size, restoration_loss.item(), recon_loss.item(), hr_warping_loss.item(), lr_warping_loss.item(), flow_loss.item(), D_TA_loss.item(), R_TA_loss.item(), total_loss.item())
                    report.update_recon_metric(result_dict['recon'], lr_blur_seq[:, :, t // 2, :, :])
                    report.update_recon_metric(result_dict['output'], hr_sharp_seq[:, :, t//2, :, :])

        period_time = time.time() - start
        prefix_str = f'[{epoch}/{self.config.num_epochs}]\t'
        result_str = report.val_result_str(period_time)

        val_log.write(prefix_str + result_str)

        if self.config.stage == 1:
            return report.recon_psnr
        elif self.config.stage == 2:
            return report.psnr

    def test(self, dataloader):
        from utils import denorm
        self.model.eval()

        with torch.no_grad():
            for idx, (lr_blur_seq, filename) in enumerate(dataloader):
                lr_blur_seq = lr_blur_seq.cuda()

                result_dict = self.model(lr_blur_seq)
                output = result_dict['output']

                output = output.squeeze(dim=0)
                output = denorm(output)

                filename = filename[0]
                filepath = os.path.basename(os.path.dirname(filename))
                filename = os.path.basename(filename)
                filename = os.path.join(self.config.save_dir, 'test', filepath, filename)
                self.save_manager.save_image(output, filename)

    def test_quantitative_result(self, gt_dir, output_dir, image_border):
        import cv2
        import glob

        report = TestReport(output_dir)
        scene_list = sorted(glob.glob(os.path.join(gt_dir, '*')))

        for scene in scene_list:
            scene_name = os.path.basename(scene)
            filelist = sorted(glob.glob(os.path.join(scene, '*.png')))
            report.scene_init(scene_name)
            for filename in filelist[image_border:-image_border]:
                gt_img = cv2.imread(filename)
                output_img = cv2.imread(os.path.join(output_dir, scene_name, os.path.basename(filename)))
                report.update_metric(gt_img, output_img, os.path.basename(filename))
            report.scene_del(scene_name)