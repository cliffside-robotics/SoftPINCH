import torch


class FusionNet_train_eval():
    def train_one_epoch(self, model, train_loader, criterion, optimizer, device):
        '''
        Docstring for train_one_epoch

        :param model: Provide the model to be trained
        :param train_loader: Provide the dataloader for training
        :param criterion: Loss function
        :param optimizer: Optimizer for training
        :param device: Device to run the training on (CPU or GPU)

        :return avg_loss : float
        '''
        # Make sure gradient tracking is on, and do a pass over the data
        model.train()

        running_loss = 0.0

        for eeg, emg, eeg_lab, emg_lab in train_loader:
            # Every data instance is an input + label pair
            X_eeg, X_emg, y_eeg, y_emg = eeg.to(device), emg.to(device), eeg_lab.to(device), emg_lab.to(device)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            final_logits, eeg_logits, emg_logits = model(eeg = X_eeg, emg = X_emg)

            # Compute the loss and its gradients
            loss_final = criterion(final_logits, y_emg)        # EMG has all 5 lables (contract per finger, release per finger, rest all fingers)
            loss_eeg   = criterion(eeg_logits, y_eeg)          # EEG has only 3 lables (contract, release, rest)
            loss_emg   = criterion(emg_logits, y_emg)   

            loss = loss_final + 0.3 * loss_eeg + 0.3 * loss_emg
            
            loss.backward()                         # Backward pass

            # Adjust learning weights
            optimizer.step()

            # Metrics
            running_loss += loss.item()
                
        avg_loss = running_loss / len(train_loader) # loss per batch

        return avg_loss

    def validate_one_epoch(self, model, val_loader, criterion, device):
        '''
        Docstring for evaluate_one_epoch

        :param model: Provide the model to be evaluated
        :param val_loader: Provide the dataloader for testing
        :param criterion: Loss function
        :param device: Device to run the evaluation on (CPU or GPU)

        :return avg_vloss : float, vacc : float, H : list [batch, h_state], Y : list [batch]
        '''
        running_vloss = 0.0
        vcorrect = 0
        vtotal = 0

        # Set the model to evaluation mode
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for eeg, emg, eeg_lab, emg_lab in val_loader:
                X_veeg, X_vemg, y_veeg, y_vemg = eeg.to(device), emg.to(device), eeg_lab.to(device), emg_lab.to(device)

                # Forward pass: compute predicted outputs by passing inputs to the model
                final_vlogits, eeg_vlogits, emg_vlogits = model(eeg = X_veeg, emg = X_vemg)
   
                # Compute the loss and its gradients
                loss_final = criterion(final_vlogits, y_vemg)        # EMG has all 5 lables (contract per finger, release per finger, rest all fingers)
                loss_eeg   = criterion(eeg_vlogits, y_veeg)          # EEG has only 3 lables (contract, release, rest)
                loss_emg   = criterion(emg_vlogits, y_vemg)  

                vloss = loss_final + 0.3 * loss_eeg + 0.3 * loss_emg

                # Update running validation loss
                running_vloss += vloss.item()
                
                # Predicted class index
                _, vpredicted = torch.max(final_vlogits, 1)
                
                # Accuracy statistics
                vtotal += y_vemg.size(0)
                vcorrect += (vpredicted == y_vemg).sum().item()

        avg_vloss = running_vloss / len(val_loader) # loss per batch
        vacc = 100 * vcorrect / vtotal
        return avg_vloss, vacc, None
    
    def inference_one_epoch(self, model, test_loader, criterion, device):
        """
        Final model evaluation on the TEST dataset.

        IMPORTANT:
        - This function must be called ONLY once after training
        and hyperparameter optimization are finished.
        - No model updates occur here.
        - Used to report final thesis performance.

        Parameters
        ----------
        model : torch.nn.Module
            Trained model to evaluate.

        test_loader : DataLoader
            DataLoader containing ONLY unseen test data.

        criterion : loss function
            Same loss used during training.

        device : torch.device
            CPU or GPU.

        Returns
        -------
        avg_test_loss : float
            Average loss across all batches.

        test_acc : float
            Classification accuracy in percentage.

        all_preds : np.ndarray
            Predicted class labels.

        all_labels : np.ndarray
            Ground truth labels.
        """

        # Put model in evaluation mode
        model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        # Store predictions for later analysis
        all_preds = []
        all_labels = []

        # Disable gradient computation (saves memory + faster)
        with torch.no_grad():

            for eeg, emg, eeg_lab, emg_lab in test_loader:
                X_eeg, X_emg, y_eeg, y_emg = eeg.to(device), emg.to(device), eeg_lab.to(device), emg_lab.to(device)

                # Forward pass
                final_logits, eeg_logits, emg_logits = model(eeg = X_eeg, emg = X_emg)

                # Compute the loss and its gradients
                loss_final = criterion(final_logits, y_emg)        # EMG has all 5 lables (contract per finger, release per finger, rest all fingers)
                loss_eeg   = criterion(eeg_logits, y_eeg)          # EEG has only 3 lables (contract, release, rest)
                loss_emg   = criterion(emg_logits, y_emg)  

                loss = loss_final + 0.3 * loss_eeg + 0.3 * loss_emg

                # Update running validation loss
                running_loss += loss.item()

                # Predicted class index
                _, predicted = torch.max(final_logits, dim=1)             # index of max value (predicted class)

                # Accuracy statistics
                total += y_emg.size(0)
                correct += (predicted == y_emg).sum().item()       # Sum of correct labels equal to predictions

                # Store outputs for confusion matrix etc.
                all_preds.append(predicted.cpu())
                all_labels.append(y_emg.cpu())

        # -----------------------------------
        # Final metrics
        # -----------------------------------
        avg_test_loss = running_loss / len(test_loader)
        test_acc = 100 * correct / total

        # Concatenate stored tensors
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        return avg_test_loss, test_acc, all_preds, all_labels
    
class SingleNet_train_eval():
    def train_one_epoch(self, model, train_loader, criterion, optimizer, device):
        '''
        Docstring for train_one_epoch

        :param model: Provide the model to be trained
        :param train_loader: Provide the dataloader for training
        :param criterion: Loss function
        :param optimizer: Optimizer for training
        :param device: Device to run the training on (CPU or GPU)

        :return avg_loss : float
        '''
        
        # Make sure gradient tracking is on, and do a pass over the data
        model.train()

        running_loss = 0.0

        for inp, lab in train_loader:
            # Every data instance is an input + label pair
            inputs, labels = inp.to(device), lab.to(device)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            logits, _, _ = model(inputs)

            # Compute the loss and its gradients
            loss = criterion(logits, labels)
            loss.backward()                         # Backward pass

            # Adjust learning weights
            optimizer.step()

            # Metrics
            running_loss += loss.item()
                
        avg_loss = running_loss / len(train_loader) # loss per batch

        return avg_loss

    def validate_one_epoch(self, model, val_loader, criterion, device):
        '''
        Docstring for evaluate_one_epoch

        :param model: Provide the model to be evaluated
        :param val_loader: Provide the dataloader for testing
        :param criterion: Loss function
        :param device: Device to run the evaluation on (CPU or GPU)

        :return avg_vloss : float, vacc : float, H : list [batch, h_state], Y : list [batch]
        '''
        running_vloss = 0.0
        vcorrect = 0
        vtotal = 0

        # Set the model to evaluation mode
        model.eval()

        # L_list = []
        # C_list = []
        # W_list = []
        # Y_list = []

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for inp, lab in val_loader:
                vinputs, vlabels = inp.to(device), lab.to(device)
                # Forward pass: compute predicted outputs by passing inputs to the model
                vlogits, _, _ = model(vinputs)

                # L_list.append(vlogits.cpu())
                # C_list.append(context.cpu())
                # W_list.append(attn_weight.cpu())
                # Y_list.append(vlabels.cpu())

                # Calculate the loss
                vloss = criterion(vlogits, vlabels)

                # Update running validation loss
                running_vloss += vloss.item()
                
                _, vpredicted = torch.max(vlogits, 1)
                vtotal += vlabels.size(0)
                vcorrect += (vpredicted == vlabels).sum().item()

        avg_vloss = running_vloss / len(val_loader) # loss per batch
        vacc = 100 * vcorrect / vtotal
        return avg_vloss, vacc, None#[L_list, C_list, W_list, Y_list]

    def inference_one_epoch(self, model, test_loader, criterion, device):
        """
        Final model evaluation on the TEST dataset.

        IMPORTANT:
        - This function must be called ONLY once after training
        and hyperparameter optimization are finished.
        - No model updates occur here.
        - Used to report final thesis performance.

        Parameters
        ----------
        model : torch.nn.Module
            Trained model to evaluate.

        test_loader : DataLoader
            DataLoader containing ONLY unseen test data.

        criterion : loss function
            Same loss used during training.

        device : torch.device
            CPU or GPU.

        Returns
        -------
        avg_test_loss : float
            Average loss across all batches.

        test_acc : float
            Classification accuracy in percentage.

        all_preds : np.ndarray
            Predicted class labels.

        all_labels : np.ndarray
            Ground truth labels.
        """

        # Put model in evaluation mode
        model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        # Store predictions for later analysis
        all_preds = []
        all_labels = []

        # Disable gradient computation (saves memory + faster)
        with torch.no_grad():

            for inputs, labels in test_loader:

                # Move batch to device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                logits, _, _ = model(inputs)

                # Compute loss
                loss = criterion(logits, labels)                    # Computes negative log likelihood
                running_loss += loss.item()

                # Predicted class index
                _, predicted = torch.max(logits, dim=1)             # index of max value (predicted class)

                # Accuracy statistics
                total += labels.size(0)
                correct += (predicted == labels).sum().item()       # Sum of correct labels equal to predictions

                # Store outputs for confusion matrix etc.
                all_preds.append(predicted.cpu())
                all_labels.append(labels.cpu())

        # -----------------------------------
        # Final metrics
        # -----------------------------------
        avg_test_loss = running_loss / len(test_loader)
        test_acc = 100 * correct / total

        # Concatenate stored tensors
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        return avg_test_loss, test_acc, all_preds, all_labels