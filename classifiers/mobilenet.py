#Inspired by https://github.com/fschmid56/EfficientAT

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from sklearn import metrics

from features import AugmentMelSTFT
from classifiers.mobilenet_utils import NAME_TO_WIDTH
from classifiers.mobilenet_utils import get_model as get_mobilenet
from classifiers.mobilenet_utils import exp_warmup_linear_down
from classifiers.mobilenet_utils import mixup

from utils import show_scores

from scipy.special import softmax

class MobileNet():
    def __init__(self, 
                 cuda=False, 
                 n_mels=128,
                 resample_rate=44100,
                 window_size=800,
                 hop_size=320,
                 n_fft=1024,
                 freqm=0,
                 timem=0,
                 fmin=0,
                 fmax=None,
                 fmin_aug_range=10,
                 fmax_aug_range=2000,
                 model_name="mn05_as",
                 pretrained=True,
                 model_width=1.0,
                 head_type="mlp",
                 se_dims="c",
                 lr=6e-5,
                 weight_decay=0.0,
                 warm_up_len=10,
                 ramp_down_len=65,
                 ramp_down_start=10,
                 last_lr_value=0.01,
                 n_epochs=20,
                 mixup_alpha=0.3):
        self.cuda = cuda
        self.n_mels = n_mels
        self.resample_rate = resample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.n_fft = n_fft
        self.freqm = freqm
        self.timem = timem
        self.fmin = fmin
        self.fmax = fmax
        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range

        self.model_name = model_name
        self.pretrained = pretrained

        self.model_width = model_width

        self.head_type = head_type
        self.se_dims = se_dims
    
        self.lr = lr
        self.weight_decay = weight_decay
        self.warm_up_len = warm_up_len
        self.ramp_down_len =ramp_down_len
        self.ramp_down_start = ramp_down_start
        self.last_lr_value = last_lr_value

        self.n_epochs = n_epochs

        self.mixup_alpha = mixup_alpha
        

    def _mel_forward(self, x, mel):
        old_shape = x.size()
        x = x.reshape(-1, old_shape[1])
        x = mel(x)
        x = x.reshape(old_shape[0], 1, x.shape[1], x.shape[2])
        return x

    
    def _test(self, model, mel, eval_loader, device, showPlots=True):
        model.eval()
        mel.eval()

        targets = []
        outputs = []
        losses = []
        for batch in eval_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                x = self._mel_forward(x, mel)
                y_hat, _ = model(x)
            targets.append(y.cpu().numpy())
            outputs.append(y_hat.float().cpu().numpy())
            losses.append(F.cross_entropy(y_hat, y).cpu().numpy())

        targets = np.concatenate(targets)
        outputs = np.concatenate(outputs)
        outputs = softmax(outputs, axis=1)[:,1]
        losses = np.stack(losses)

        show_scores(targets, outputs, csvfile=None, showPlots=showPlots)

        return losses.mean()


    def train(self, train_loader, val_loader):
        # Train Models for Acoustic Scene Classification

        device = torch.device('cuda') if self.cuda and torch.cuda.is_available() else torch.device('cpu')

        # model to preprocess waveform into mel spectrograms
        self.mel = AugmentMelSTFT(n_mels=self.n_mels,
                            sr=self.resample_rate,
                            win_length=self.window_size,
                            hopsize=self.hop_size,
                            n_fft=self.n_fft,
                            freqm=self.freqm,
                            timem=self.timem,
                            fmin=self.fmin,
                            fmax=self.fmax,
                            fmin_aug_range=self.fmin_aug_range,
                            fmax_aug_range=self.fmax_aug_range
                            )
        self.mel.to(device)

        # load prediction model
        model_name = self.model_name
        pretrained_name = model_name if self.pretrained else None
        width = NAME_TO_WIDTH(model_name) if model_name and self.pretrained else self.model_width
        
        self.model = get_mobilenet(width_mult=width, pretrained_name=pretrained_name,
                            head_type=self.head_type, se_dims=self.se_dims,
                            num_classes=2)
        self.model.to(device)

        # dataloader
        """
        dl = DataLoader(dataset=get_training_set(resample_rate=args.resample_rate,
                                                roll=False if args.no_roll else True,
                                                wavmix=False if args.no_wavmix else True,
                                                gain_augment=args.gain_augment,
                                                fold=args.fold),
                        worker_init_fn=worker_init_fn,
                        num_workers=args.num_workers,
                        batch_size=args.batch_size,
                        shuffle=True)
        """
        dl = train_loader

        # evaluation loader
        """
        eval_dl = DataLoader(dataset=get_test_set(resample_rate=args.resample_rate, fold=args.fold),
                            worker_init_fn=worker_init_fn,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size)
        """
        eval_dl = val_loader

        # optimizer & scheduler
        lr = self.lr
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # phases of lr schedule: exponential increase, constant lr, linear decrease, fine-tune
        schedule_lambda = \
            exp_warmup_linear_down(self.warm_up_len, self.ramp_down_len, self.ramp_down_start, self.last_lr_value)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)

        name = None
        accuracy, val_loss = float('NaN'), float('NaN')

        for epoch in range(self.n_epochs):
            self.mel.train()
            self.model.train()
            train_stats = dict(train_loss=list())
            print("")
            print("")
            print("==== Epoch {}/{} ====".format(epoch + 1, self.n_epochs))
            #print("Epoch {}/{}: accuracy: {:.4f}, val_loss: {:.4f}"
            #  
            #                   .format(epoch + 1, self.n_epochs, accuracy, val_loss))
            print("  Number of batch:", len(dl))
            iBatch = 0
            y_train_pred = []
            y_train_real = []
            for batch in dl:
                print(iBatch, end="...")
                iBatch += 1
                x, y = batch
                bs = x.size(0)
                x, y = x.to(device), y.to(device)
                x = self._mel_forward(x, self.mel)

                if self.mixup_alpha:
                    rn_indices, lam = mixup(bs, self.mixup_alpha)
                    lam = lam.to(x.device)
                    x = x * lam.reshape(bs, 1, 1, 1) + \
                        x[rn_indices] * (1. - lam.reshape(bs, 1, 1, 1))
                    y_hat, _ = self.model(x)
                    samples_loss = (F.cross_entropy(y_hat, y, reduction="none") * lam.reshape(bs) +
                                    F.cross_entropy(y_hat, y[rn_indices], reduction="none") * (
                                                1. - lam.reshape(bs)))

                else:
                    y_hat, _ = self.model(x)
                    samples_loss = F.cross_entropy(y_hat, y, reduction="none")
                
                # loss
                loss = samples_loss.mean()

                # append training statistics
                train_stats['train_loss'].append(loss.detach().cpu().numpy())

                # Update Model
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                y_train_real.append(y.detach().cpu().numpy())
                y_train_pred.append(y_hat.detach().float().cpu().numpy())
            # Update learning rate
            scheduler.step()


            print("")
            print("")
            print("=> Epoch", (epoch+1), "Training scores:")
            y_train_pred = np.concatenate(y_train_pred)
            y_train_real = np.concatenate(y_train_real)
            y_train_pred = softmax(y_train_pred, axis=1)[:,1]

            showPlots = False
            if epoch+1 == self.n_epochs:
                showPlots = True
            show_scores(y_train_real, y_train_pred, csvfile=None, showPlots=showPlots)
            print("train_loss", np.mean(train_stats['train_loss']))

            print("")
            print("=> Epoch", (epoch+1), "Validation scores:")
            val_loss = self._test(self.model, self.mel, eval_dl, device, showPlots)

            # log train and validation statistics
            print("val_loss", val_loss)
            """
            wandb.log({"train_loss": np.mean(train_stats['train_loss']),
                   "accuracy": accuracy,
                   "val_loss": val_loss
                   })
                   """

            # remove previous model (we try to not flood your hard disk) and save latest model
            #if name is not None:
            #    os.remove(os.path.join(wandb.run.dir, name))
            #name = f"mn{str(width).replace('.', '')}_esc50_epoch_{epoch}_mAP_{int(round(accuracy*100))}.pt"
            #torch.save(model.state_dict(), os.path.join(wandb.run.dir, name))

    def test(self, test_loader):
        
        device = torch.device('cuda') if self.cuda and torch.cuda.is_available() else torch.device('cpu')

        loss = self._test(self.model, self.mel, test_loader, device, True)