import torch
import time
import copy


def trainB(model, dataloaders, criterion, optimizer, scheduler, save_path, device, num_epochs=100):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        since1 = time.time()
        
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  
            else:
                model.eval() 

            running_loss = 0.0
            num = 0
            iter1 = 0
            
            for encoder_input, decoder_input in zip(dataloaders[phase][0], dataloaders[phase][1]):
                encoder_input = encoder_input.to(device)
                decoder_input = decoder_input.to(device)
                
                num += 1
                iter1 += 1
                
                GT = decoder_input[:,-1,0]  
                decoder_input = decoder_input[:,-1,1:]  
                                
                optimizer.zero_grad()
                
                if iter1%50 == 0:print('iter : {}'.format(iter1))

                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(encoder_input, decoder_input)
                    loss = criterion(outputs, GT)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * outputs.size(0)

            total_num = dataloaders[phase][0].batch_size * num
            epoch_loss = running_loss / total_num
            time_elapsed = time.time() - since1

            print('{} Loss: {:.4f} Time: {:.0f}m {:.0f}s'.format(
                phase, epoch_loss, time_elapsed//60, time_elapsed%60))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss :
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, save_path+"/epoch_{}_torch_model".format(epoch+1))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def trainA(model, dataloaders, criterion, optimizer, scheduler, save_path, device, num_epochs=100):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        since1 = time.time()
        
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()  

            running_loss = 0.0
            num = 0
            iter1 = 0
            
            # data : (batch, sequence<144>, dim)
            # GT : (batch, 1, dim)
            for data, GT in zip(dataloaders[phase][0], dataloaders[phase][1]):
                data = data.to(device)
                GT = GT.to(device)
                
                num += 1
                iter1 += 1
                
                GT = torch.squeeze(GT, 1)[:,0]
                
                optimizer.zero_grad()
                
                if iter1%50 == 0:print('iter : {}'.format(iter1))

                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(data)
                    loss = criterion(outputs, GT)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * outputs.size(0)

            total_num = dataloaders[phase][0].batch_size * num
            epoch_loss = running_loss / total_num
            time_elapsed = time.time() - since1

            print('{} Loss: {:.4f} Time: {:.0f}m {:.0f}s'.format(
                phase, epoch_loss, time_elapsed//60, time_elapsed%60))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss :
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, save_path+"/epoch_{}_torch_model".format(epoch+1))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model