import numpy as np
import torch


from datasets import load_metric
cer_metric = load_metric("cer")
from tokenizers import Tokenizer
tokenizer = processor.tokenizer
from tqdm.notebook import tqdm

#tokenizer.convert_tokens_to_ids
def compute_cer(pred_ids, label_ids):
    pred_str = tokenizer.batch_decode(pred_ids.tolist(), skip_special_tokens=True)
    label_ids[label_ids == -100] = tokenizer.convert_tokens_to_ids("[PAD]")
    label_str = tokenizer.batch_decode(label_ids.tolist(), skip_special_tokens=True)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return cer


def train_epoch(model, train_dataloader, optimizer, device):
    # train
       model.train()
       train_loss = 0.0
       valid_cer = 0.0
       lss_history = []
       cer_hist = []
       for i, batch in enumerate(tqdm(train_dataloader)):
          # get the inputs
          for k,v in batch.items():
            batch[k] = v.to(device)

          # forward + backward + optimize
          outputs = model(**batch)
          loss = outputs.loss
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()

          train_loss += loss.item()

          # acc calculations
          lss_history.append(loss.item())
          #acc.append(((pred.argmax(axis = 1) == labels).type(torch.float)).mean().item())
          #if i % 100 == 0: print(f"Loss: {loss.item()}") 
          with torch.no_grad():
              cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"])
              #valid_cer += cer
              cer_hist.append(cer)

       #print(f"Loss after epoch {epoch}:", train_loss/len(train_dataloader))
       #model.save_pretrained(f"version_{version}/epoch_{epoch}")
       return np.mean(lss_history) , np.mean(cer_hist)


def validate_epoch(model, train_dataloader, optimizer, device):
    model.eval()
    train_loss = 0.0
    cer_hist = []
    lss_history = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(train_dataloader)):
            # get the inputs
            for k,v in batch.items():
                batch[k] = v.to(device)

            # forward + backward + optimize
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

            # acc calculations
            lss_history.append(loss.item())
            cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"])
            cer_hist.append(cer)
            #acc.append(((pred.argmax(axis = 1) == labels).type(torch.float)).mean().item())
            #if i % 100 == 0: print(f"Loss: {loss.item()}") 

    #print(f"Loss after epoch {epoch}:", train_loss/len(train_dataloader))
    #model.save_pretrained(f"version_{version}/epoch_{epoch}")
    return np.mean(lss_history) , np.mean(cer_hist)


def train_epoch_old(model,dataloader,loss,optimizer,device):
    model.train()
    acc = []
    lss_history = []
    for _ , (data,labels) in enumerate(dataloader):
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        pred = model(data)
        lss = loss(pred,labels)

        lss.backward()
        optimizer.step()


        # acc calculations
        lss_history.append(lss.item())
        acc.append(((pred.argmax(axis = 1) == labels).type(torch.float)).mean().item())


    return np.mean(lss_history) ,np.mean(acc)

def validate_epoch_old(model,dataloader,loss,device):
    model.eval()
    acc = []
    lss_history = []
    with torch.no_grad():
        for i , (data,labels) in enumerate(dataloader):
            data = data.to(device)
            labels = labels.to(device)

            pred = model(data)
            lss = loss(pred,labels)
            lss_history.append(lss.item())
            acc.append(((pred.argmax(axis = 1) == labels).type(torch.float)).mean().item())
    return np.mean(lss_history) ,np.mean(acc)



def tune_model(num_epochs,model,train_dataloader_,test_dataloader_,\
               optimizer,device,scheduler=None,earlystopping=None) :
    '''
    NOTE that the scheduler here takes the update after every epoch not step
    '''

    hist = {'train_loss': [],
            'train_acc':[],
            'test_loss': [],
            'test_acc':[]}

    last_lr = optimizer.state_dict()['param_groups'][0]['lr']
    f= 0
    for e in range(num_epochs):
        lss,acc= train_epoch(model, train_dataloader_, optimizer, device)
        test_lss,test_acc= validate_epoch(model, test_dataloader_, optimizer, device)


        if (e + 1) % 5==0:
            print(f"For epoch {e:3d} || Training Loss {lss:5.3f} || acc {acc:5.3f}",end='')
            print(f" || Testing Loss {test_lss:5.3f} || Test acc {test_acc:5.3f}")
        hist['train_loss'].append(lss)
        hist['train_acc'].append(acc)
        hist['test_loss'].append(test_lss)
        hist['test_acc'].append(test_acc)



        #
        if earlystopping:
            if earlystopping(model,test_lss): # should terminate
                print('Early Stopping Activated')
                return hist
        # if you have scheduler
        if scheduler:
            scheduler.step(test_lss)
            try:
        # applying manual verbose for the scheduler
                if last_lr != scheduler.get_last_lr()[0]:
                    print(f'scheduler update at Epoch {e+1}')
                    last_lr = scheduler.get_last_lr()[0]
            except:
                f+=1
    return hist
