from utils.loss import recon_loss, sup_loss,AverageMeter
import time
import torch


def train(encoder,decoder,Q,opt,data_loader,is_wandb=False,verbose_freq = 500,is_verbose = False):
    encoder.train()
    decoder.train()
    Q.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_losses = AverageMeter()
    end = time.time()
    
    for batch_idx, (x,t,m,y,y_mask ) in enumerate(data_loader): 
        data_time.update(time.time() - end)
        
        opt.zero_grad()
        c_hat = encoder(x*m)
        x_hat = decoder(c_hat)
        y_hat = Q(c_hat)
        
        loss = recon_loss(x_hat,x,m) + sup_loss(y_hat,y,y_mask)
        
        loss.backward()
        opt.step()
        
        train_losses.update(loss.data.item())
        batch_time.update(time.time() - end)
        end = time.time()
        
        if is_verbose:
            if (batch_idx+1) % verbose_freq == 0:
                msg = 'Epoch: [{0}/{3}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {train_loss.val:.5f} ({train_loss.avg:.5f})\t'.format(
                          epoch_idx+1, batch_idx,len(data_loader), n_epochs , batch_time=batch_time,
                          speed=x.size(0)/batch_time.val,
                          data_time=data_time, train_loss=train_losses)
                print(msg)
        
        if is_wandb:
            wandb.log({"batch_loss": loss.data.item()})
    
    
    if is_wandb:
            wandb.log({"train_epoch_loss": train_losses.avg})
            wandb.log({"training time/Iter": batch_time.sum/len(data_loader)})
    
    
    # print(f"Evaluation   Epoch : {epoch_idx+1}  =====================>")
    if is_verbose:
        print(f"Training Epoch Loss: {train_losses.avg}")
        
        
        
def validate(encoder,decoder,Q,data_loader,is_wandb=False,is_verbose = False):
    encoder.eval()
    decoder.eval()
    Q.eval()
    
    val_losses = AverageMeter()
    
    
    with torch.no_grad():
        for batch_idx, (x,t,m,y,y_mask) in enumerate(data_loader): 
            c_hat = encoder(x*m)
            x_hat = decoder(c_hat)
            y_hat = Q(c_hat)

            loss = recon_loss(x_hat,x,m) + sup_loss(y_hat,y,y_mask)
            
            val_losses.update(loss.data.item())
    
    if is_wandb:
            wandb.log({"val_epoch_loss": val_losses.avg})
    
    
    # print(f"Evaluation   Epoch : {epoch_idx+1}  =====================>")
    if is_verbose:
        print(f"Val Epoch Loss: {val_losses.avg}")
    return val_losses.avg
          