import numpy as np
from numpy import save, load
import pickle

from pathlib import Path 

from nilearn.connectome import ConnectivityMeasure
from nilearn.connectome import sym_matrix_to_vec

import torch
from torch import nn
import torch.nn.init as init
from sklearn.model_selection import train_test_split, KFold
#from torchinfo import summary
from torch.utils.data import DataLoader
from helper_functions import accuracy_fn
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import warnings

device = "cuda" if torch.cuda.is_available() else "cpu"

class NoiseLayer(nn.Module):
    def __init__(self, mean=0, stddev=1):
        super(NoiseLayer, self).__init__()
        self.mean = mean
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            global noise_seed
            noise_seed += 1
            torch.manual_seed(noise_seed)
            #Gaussian Noise
            noise = torch.randn_like(x) * self.stddev + self.mean
            return x + noise
        return x

def initialize_weights(module):
    if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
        init.xavier_normal_(module.weight)
        if module.bias is not None:
            init.constant_(module.bias, 0)
        
class Stacked3CNNAutoencoder(nn.Module):
    def __init__(self, num_kernels: int, dropout: int, mean: int, stddev: int):
        super().__init__()

        self.mean = mean
        self.stddev = stddev
        self.num_kernels = num_kernels

        self.model1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=num_kernels, kernel_size = (1,392)),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = (392,1))
        )

        self.model4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=num_kernels, kernel_size = (4,392)),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = (389,1))
        )

        self.model7 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=num_kernels, kernel_size = (7,392)),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = (386,1))
        )
        
        self.dropout2d = nn.Dropout2d(dropout)
        self.dropout = nn.Dropout(dropout)

        self.tanh = nn.Tanh()
        
        #self.fc = nn.Linear(in_features=3*num_kernels, out_features=1)
        self.fc1 = nn.Linear(in_features=3*num_kernels, out_features=num_kernels)
        self.fc = nn.Linear(in_features=num_kernels, out_features=1)
        
        #self.bottleneck = nn.Linear(in_features=3*num_kernels, out_features=num_kernels)
        #self.reconfc = nn.Linear(in_features=num_kernels, out_features=3*num_kernels)
        
        #self.reconfc1 = nn.Linear(in_features=num_kernels, out_features=3*num_kernels)
        
        self.deconv1 = nn.Sequential(
            nn.Upsample(scale_factor=(392,1), mode='nearest'),
            nn.ConvTranspose2d(in_channels=num_kernels, out_channels=1, kernel_size=(1,392)),
            nn.Tanh()
        )
        
        self.deconv4 = nn.Sequential(
            nn.Upsample(scale_factor=(389,1), mode='nearest'),
            nn.ConvTranspose2d(in_channels=num_kernels, out_channels=1, kernel_size=(4,392)),
            nn.Tanh()
        )
        
        self.deconv7 = nn.Sequential(
            nn.Upsample(scale_factor=(386,1), mode='nearest'),
            nn.ConvTranspose2d(in_channels=num_kernels, out_channels=1, kernel_size=(7,392)),
            nn.Tanh()
        )

        self.apply(initialize_weights)
        
    def forward(self, x, inference_mode):
        if not inference_mode:
            x = NoiseLayer(mean=self.mean, stddev=self.stddev)(x)
        
        model1 = self.model1(x)
        model4 = self.model4(x)
        model7 = self.model7(x)
        
        model = torch.cat((model1, model4, model7), dim=2)
        
        #model = self.dropout2d(model)
        model = torch.flatten(model, start_dim=1)
        model = self.tanh(self.fc1(model))
        model = self.dropout(model)

        classification = torch.sigmoid(self.fc(model))

        #model = self.reconfc1(model)

        #model = model.view(x.size(0), self.num_kernels, 3, 1)
        #model1, model4, model7 = torch.split(model, split_size_or_sections=[1, 1, 1], dim=2)
        
        model1 = self.deconv1(model1)
        model4 = self.deconv4(model4)
        model7 = self.deconv7(model7)
        
        reconstruction = (model1 + model4 + model7) / 3
        #reconstruction = torch.zeros(2)
        return classification, reconstruction


def l1_regularization(model, lambda_l1):
    l1_reg = 0.0
    for layer, param in model.named_parameters():
        l1_reg += torch.sum(torch.abs(param))
    return lambda_l1 * l1_reg

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100

def sensitivity_fn(y_true, y_pred):
    TP = ((y_true == 1) & (y_pred == 1)).sum().item()
    FN = ((y_true == 1) & (y_pred == 0)).sum().item()
    return TP/(TP+FN) * 100 if (TP+FN) != 0 else 0

def specificity_fn(y_true, y_pred):
    TN = ((y_true == 0) & (y_pred == 0)).sum().item()
    FP = ((y_true == 0) & (y_pred == 1)).sum().item()
    return TN/(TN+FP) * 100 if (TN+FP) != 0 else 0


def training(folds, train_dataloader, test_dataloader, val_dataloader, epochs, alpha, beta, kernels, dropout, batch_size, mean, stddev, fold, learning_rate, l2, save_model=False):
    training_accuracies = list()
    train_accuracies = list()
    validation_accuracies = list()
    kfold_validation_accuracies = [[], [], [], [], [], [], [], [], [], []]

    training_losses = list()

    training_classification_losses = list()
    train_classification_losses = list()
    validation_classification_losses = list()
    kfold_validation_classification_losses = [[], [], [], [], [], [], [], [], [], []]

    training_reconstruction_losses = list()
    train_reconstruction_losses = list()
    validation_reconstruction_losses = list()
    
    validation_y = list()
    
    #Defining Model
    model = Stacked3CNNAutoencoder(num_kernels=kernels, dropout=dropout, mean=mean, stddev=stddev)
    #print(summary(model_0, input_size=(32, 1, 392, 392), col_names=["input_size", "output_size", "num_params", "trainable"], col_width=20, row_settings=["var_names"]))

    model = model.to(device)
    model.train()

    class_loss_fn = nn.BCELoss()
    recon_loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2)

    for epoch in range(epochs):
        training_loss = 0
        classification_loss = 0
        reconstruction_loss = 0
        training_accuracy = 0
        print(f"\n=============================================Epoch: {epoch + 1}=============================================")
        for batch_idx, (X, y) in enumerate(train_dataloader):
            
            X = X.to(device)
            y = y.to(device)
            
            y_pred, X_pred = model(X, inference_mode=False)

            class_loss = class_loss_fn(y_pred, y)
            recon_loss = recon_loss_fn(X_pred, X)
            
            #Delete the below line for the Autoencoder
            #recon_loss = recon_loss_fn(X_pred, X_pred)
            
            normalized_class_loss = class_loss  / (class_loss + recon_loss)
            normalized_recon_loss = recon_loss / (class_loss + recon_loss)
            
            loss = alpha * class_loss + beta * recon_loss
            
            classification_loss += class_loss.item()
            reconstruction_loss += recon_loss.item()
            
            training_loss += loss.item()

            training_accuracy += accuracy_fn(y_true=y, y_pred=(y_pred>0.5).type(torch.float32))

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if batch_idx % 16 == 0:
                print(f"Looked at {batch_idx * len(X)}/{len(train_dataloader.dataset)} samples")

        classification_loss /= len(train_dataloader)
        reconstruction_loss /= len(train_dataloader)
        training_loss /= len(train_dataloader)
                
        print(f"\nClassification loss: {classification_loss:.5f}\nReconstruction loss: {reconstruction_loss:.5f}\nTraining loss: {training_loss:.5f}\nTraining Accuracy {training_accuracy:.2f}%")
        
        training_accuracy /= len(train_dataloader)
        
        training_accuracies.append(training_accuracy)
        training_classification_losses.append(classification_loss)
        training_reconstruction_losses.append(reconstruction_loss)
        training_losses.append(training_loss)

        model.eval()

        with torch.inference_mode():
            
            '''
            #Evaluating across all 10-folds
            
            for fold_i in range(len(folds)):
                val_data = TensorDataset(folds[fold_i][0], folds[fold_i][1])
                val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
                
                val_fold_class_loss = 0
                val_fold_recon_loss = 0
                val_fold_accuracy = 0

                for X, y in val_loader:
                    X=X.to(device)
                    y=y.to(device)
                    y_val_pred, _ = model(X, inference_mode=True)
                    val_fold_class_loss += class_loss_fn(y_val_pred, y).item()
                    val_fold_accuracy += accuracy_fn(y_true=y, y_pred=(y_val_pred>0.5).type(torch.float32))

                val_fold_class_loss /= len(val_loader)
                val_fold_accuracy /= len(val_loader)
                
                kfold_validation_accuracies[fold_i].append(val_fold_accuracy)
                kfold_validation_classification_losses[fold_i].append(val_fold_class_loss)
            '''

            #Evaluating on train set
            
            train_class_loss = 0
            train_recon_loss = 0
            train_accuracy = 0
            train_sensitivity = 0
            train_specificity = 0
            
            for X, y in train_dataloader:
                X=X.to(device)
                y=y.to(device)
                y_train_pred, X_train_pred = model(X, inference_mode=True)
                train_class_loss += class_loss_fn(y_train_pred, y).item()
                train_recon_loss += recon_loss_fn(X_train_pred, X).item()
                #Delete before running the autoencoder
                #train_recon_loss += recon_loss_fn(X, X).item()
                y_train_pred_binary = (y_train_pred>0.5).type(torch.float32)
                train_accuracy += accuracy_fn(y_true=y, y_pred=y_train_pred_binary)
                train_sensitivity += sensitivity_fn(y_true=y, y_pred=y_train_pred_binary)
                train_specificity += specificity_fn(y_true=y, y_pred=y_train_pred_binary)

            train_class_loss /= len(train_dataloader)
            train_recon_loss /= len(train_dataloader)
            train_accuracy /= len(train_dataloader)
            train_sensitivity /= len(train_dataloader)
            train_specificity /= len(train_dataloader)
            
            train_accuracies.append(train_accuracy)
            train_classification_losses.append(train_class_loss)
            train_reconstruction_losses.append(train_recon_loss)
            
            #Evaluating on validation set
            
            val_class_loss = 0
            val_recon_loss = 0
            val_accuracy = 0
            val_sensitivity = 0
            val_specificity = 0
            
            for X, y in val_dataloader:
                X=X.to(device)
                y=y.to(device)
                y_val_pred, X_val_pred = model(X, inference_mode=True)
                y_pred_binary = (y_val_pred>0.5).type(torch.float32)
                
                val_class_loss += class_loss_fn(y_val_pred, y).item()
                val_recon_loss += recon_loss_fn(X_val_pred, X) .item()
                #Delete for running the autoendoer
                #val_recon_loss += recon_loss_fn(X, X).item()
                val_accuracy += accuracy_fn(y_true=y, y_pred=y_pred_binary)
                val_sensitivity += sensitivity_fn(y_true=y, y_pred=y_pred_binary)
                val_specificity += specificity_fn(y_true=y, y_pred=y_pred_binary)
                
            validation_y.append({'y':y, 'y_val_pred':y_val_pred})

            val_class_loss /= len(val_dataloader)
            val_recon_loss /= len(val_dataloader)
            val_accuracy /= len(val_dataloader)
            val_sensitivity /= len(val_dataloader)
            val_specificity /= len(val_dataloader)
            
            validation_accuracies.append(val_accuracy)
            validation_classification_losses.append(val_class_loss)
            validation_reconstruction_losses.append(val_recon_loss)
            
            #print(f"\nTraining Accuracy {training_accuracy:.2f}% | Train loss: {train_class_loss:.5f} | Train Accuracy {train_accuracy:.2f}% | Train reconstruction loss: {train_recon_loss:.5f} \n")
            print(f"\nTrain classification loss: {train_loss:.5f} | Train reconstruction loss: {train_recon_loss:.5f}")
            print(f"\nTrain Accuracy {train_accuracy:.2f}% | Train Sensitivity: {train_sensitivity:.2f}% | Train Specificity {train_specificity:.2f}%\n")
            print(f"\nValidation loss: {val_loss:.5f} | Validation reconstruction loss: {val_recon_loss:.5f}")
            print(f"\nValidation Acuracy: {val_accuracy:.2f}% | Validation Sensitivity: {val_sensitivity:.2f}% | Validation Specificity: {val_specificity:.2f}%\n")
            
            
            accuracies = (training_accuracies, train_accuracies, validation_accuracies, kfold_validation_accuracies)
            classification_losses = (training_classification_losses, train_classification_losses, validation_classification_losses, kfold_validation_classification_losses)
            reconstruction_losses = (training_reconstruction_losses, train_reconstruction_losses, validation_reconstruction_losses)

        if save_model:
            
            target_dir_path = Path(f'SavedModels/{kernels}_kernels_{learning_rate}_lr_{dropout}_dropout_{stddev}_stddev_{timeSeriesLength}_timeSeriesLength_{l2}_rglrzn_{desc}/Fold_{fold}/')
            target_dir_path.mkdir(parents=True, exist_ok=True)
            model_name = f"Epoch_{epoch}_{train_accuracy:.2f}_{val_accuracy:.2f}.pth"
            model_save_path = target_dir_path/model_name

            torch.save(model.state_dict(), model_save_path)
            
            print(f'[INFO] Saving model to: {model_save_path}')
        
        if epoch > 15 and val_accuracy > 55:
            overfit = True
            for i in range(len(train_accuracies)-1, len(train_accuracies)-16, -1):
                if (train_accuracies[i] - validation_accuracies[i]) < 5:
                    overfit = False
            if overfit:
                print("Model Overfitted! Aboring training...\n\n")
                break
            
        if training_accuracy >= 99:
            print("Training accuracy reached 99%. Aborting training...")
            break
    return accuracies, training_losses, classification_losses, reconstruction_losses, validation_y
    
def data_augment(X, y, window_slice=False, split=False):
    
    if window_slice:
        # Augmenting the dataset by window slicing with window size of 116 and skip of 13.

        X_slice_augmented = list()
        y_slice_augmented = list()
        slice_skip = timeSeriesSkip

        for subject in range(len(X)):
            start_idx = 0
            length = len(X[subject])
            while start_idx + timeSeriesLength <= length:
                X_slice_augmented.append(X[subject][start_idx:start_idx+timeSeriesLength, :])
                y_slice_augmented.append(y[subject])
                start_idx += slice_skip

        X = np.array(X_slice_augmented)
        y = np.array(y_slice_augmented)
        
    else:
        #Trimming the time series of all sites to equal lengths and creating temoporal data.
        for subject in range(len(X)):

            length = len(X[subject])

            trimleft = int((length - timeSeriesLength) / 2)
            trimright = int(length - trimleft)

            X[subject] = X[subject][trimleft:trimright, :]
    
    if split:
        # Augmenting the dataset by splitting in 2.

        X_augmented = list()
        split_len = timeSeriesLength // 2

        for sub in range(len(X)):
            X_augmented.append(np.array(X[sub][0:split_len,:]))
            X_augmented.append(np.array(X[sub][split_len:,:]))

        X = np.array(X_augmented)

        y_augmented = []
        for i in range(len(y)):
            for _ in range(2):
                y_augmented.append(y[i])

        y = np.array(y_augmented)
        
        return X, y
    return X, y

def abide_load(band_pass_filtering, global_signal_regression):
    if band_pass_filtering and global_signal_regression:
        preprocessing_type = "filt_global"
        abide_loaded = load('abide_filt_global.npy', allow_pickle=True)
        y_loaded = load('y_filt_global.npy')
    elif band_pass_filtering and not global_signal_regression:
        preprocessing_type = "filt_noglobal"
        abide_loaded = load('abide_filt_noglobal.npy', allow_pickle=True)
        y_loaded = load('y_filt_noglobal.npy')
    elif not band_pass_filtering and global_signal_regression:
        preprocessing_type = "nofilt_global"
        abide_loaded = load('abide_nofilt_global.npy', allow_pickle=True)
        y_loaded = load('y_nofilt_global.npy')
    elif not band_pass_filtering and not global_signal_regression:
        preprocessing_type = "nofilt_noglobal"
        abide_loaded = load('abide_nofilt_noglobal.npy', allow_pickle=True)
        y_loaded = load('y_nofilt_noglobal.npy')
    return abide_loaded, y_loaded, preprocessing_type
    
def core(band_pass_filtering, global_signal_regression, X_test, y_test, epochs, alpha, beta, kernels, dropout, l2, desc, random_state, mean, stddev, slice_augment, split_augment, batch_size, folds, learning_rate=0.005, save_model=False):
    
    print(f"\n##################################### Training... #####################################\n")
    
    X, y, preprocessing_type = abide_load(band_pass_filtering, global_signal_regression)
    y[y==2] = 0

    #Excluding the data from shorter sites.
    temp = list()
    for i in range(len(X)):
        if len(X[i]) in excluding:
            temp.append(i)
    X = np.delete(X, temp)
    y = np.delete(y, temp)
    
    data_des = dict()
    for i in range(len(X)):
        if len(X[i]) in data_des:
            data_des[len(X[i])] += 1
        else:
            data_des[len(X[i])] = 1
    
    tsl_idxs = dict()
    
    for keys in data_des:
        tsl_idxs[keys] = [[], []]
    
    # Adding subjects to their respective Time Series Length key and class
    for i in range(len(X)):
        tsl_idxs[len(X[i])][y[i]].append(i)
    
    # Unskewing the data
    for tsl in tsl_idxs:
        if len(tsl_idxs[tsl][0]) < len(tsl_idxs[tsl][1]):
            tsl_idxs[tsl][1] = tsl_idxs[tsl][1][:len(tsl_idxs[tsl][0])]
        elif len(tsl_idxs[tsl][0]) > len(tsl_idxs[tsl][1]):
            tsl_idxs[tsl][0] = tsl_idxs[tsl][0][:len(tsl_idxs[tsl][1])]
    
    folds_idx = [[], [], [], [], [], [], [], [], [], []]
    ifold = 0
    for tsl in tsl_idxs:
        for sub in range(len(tsl_idxs[tsl][0])):
            for i in [0,1]:
                folds_idx[ifold].append(tsl_idxs[tsl][i][sub])
            ifold += 1
            ifold %= 10
    
    folds = list()
    for fold_idx in folds_idx:
        folds.append([X[fold_idx], y[fold_idx]])
        
    # Data augmentation
    for fold_i in range(len(folds)):
        folds[fold_i] = list(data_augment(folds[fold_i][0], folds[fold_i][1], window_slice=slice_augment, split=split_augment))
        
    # Correlation Matrices computation
    for fold in folds:
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fold[0] = np.array(conn_est.fit_transform(fold[0]))

        fold[0] = torch.from_numpy(fold[0]).type(torch.float32).unsqueeze(dim=1)
        fold[1] = torch.from_numpy(fold[1]).type(torch.float32).unsqueeze(dim=1)
    
    test_dataloader=None
    
    for fold_i in range(len(folds)):
        
        '''
        if fold_i > 5:
            pass
        else:
            continue
        if fold_i == 0 or fold_i == 2 or fold_i == 7:
            pass
        else:
            continue
        '''

        #Below code is used to verify if the component and TCN models have the same data points.
        '''
        print("==================== train ========================")
        print(folds_idx[fold_i])
        print("==================== val  ========================")
        print(val)
        print("\n\n\n")
        break
        '''

        print(f"Hyperparameters for {desc}:\n{preprocessing_type}_\n{random_state}_random_state\n{kernels}_kernels\nbatch_size_{batch_size}\n{learning_rate}_learning_rate\n{timeSeriesLength}_timeSeriesLength\n{alpha}_alpha\n{dropout}_dropout\n{l2}_L2-lambda\n{mean}_mean\n{stddev}_stddev\n{fold_i+1}th_fold\n")
        print(f"\n##################################### Fold: {fold_i+1} #####################################\n")
        
        train = folds[:fold_i] + folds[fold_i+1:]
        
        X_train = list()
        y_train = list()

        for fold_j in range(len(train)):
            for subject in range(len(train[fold_j][0])):
                X_train.append(train[fold_j][0][subject])
                y_train.append(train[fold_j][1][subject])
                
        X_train = torch.stack(X_train)
        y_train = torch.stack(y_train)
                
        train_data = TensorDataset(X_train, y_train)
        val_data = TensorDataset(folds[fold_i][0], folds[fold_i][1])
        
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)

        #Training
        accuracies, training_losses, classification_losses, reconstruction_losses, validation_y = training(folds=folds, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                                                           val_dataloader=val_dataloader, epochs=epochs, kernels=kernels, dropout=dropout,
                                                           learning_rate=learning_rate, l2=l2, save_model=save_model, mean=mean, 
                                                           stddev=stddev, batch_size=batch_size, fold=fold_i, alpha=alpha, beta=beta)
        
        
        training_accuracies, train_accuracies, validation_accuracies, kfold_validation_accuracies = accuracies
        training_classification_losses, train_classification_losses, validation_classification_losses, kfold_validation_classification_losses = classification_losses
        training_reconstruction_losses, train_reconstruction_losses, validation_reconstruction_losses = reconstruction_losses
            
        #Saving Accuracies
        target_dir_path = Path(f'SavedResults/{random_state}_random_state_{kernels}_kernels_{learning_rate}_lr_{batch_size}_batch_size_{timeSeriesLength}_timeSeriesLength_{alpha}_alpha_{dropout}_dropout_{desc}_output_{stddev}_stddev/Fold_{fold_i+1}/')
        target_dir_path.mkdir(parents=True, exist_ok=True)
        
        #Saving binaries of validation y
        variable_save_path = target_dir_path/f"Fold_{fold_i+1}_validation_y.pkl"
        with open(variable_save_path, 'wb') as f:
            pickle.dump(validation_y, f)
            
        #Saving binaries of accuracies
        variable_save_path = target_dir_path/f"Fold_{fold_i+1}_training_accuracies.pkl"
        with open(variable_save_path, 'wb') as f:
            pickle.dump(training_accuracies, f)
        
        variable_save_path = target_dir_path/f"Fold_{fold_i+1}_train_accuracies.pkl"
        with open(variable_save_path, 'wb') as f:
            pickle.dump(train_accuracies, f)
        
        variable_save_path = target_dir_path/f"Fold_{fold_i+1}_validation_accuracies.pkl"
        with open(variable_save_path, 'wb') as f:
            pickle.dump(validation_accuracies, f)
        
        variable_save_path = target_dir_path/f"Fold_{fold_i+1}_kfold_validation_accuracies.pkl"
        with open(variable_save_path, 'wb') as f:
            pickle.dump(kfold_validation_accuracies, f)
        
        #Saving Training Losses
        variable_save_path = target_dir_path/f"Fold_{fold_i+1}_training_losses.pkl"
        with open(variable_save_path, 'wb') as f:
            pickle.dump(training_losses, f)
            
        #Saving Classification Losses
        variable_save_path = target_dir_path/f"Fold_{fold_i+1}_training_classification_losses.pkl"
        with open(variable_save_path, 'wb') as f:
            pickle.dump(training_classification_losses, f)
        
        variable_save_path = target_dir_path/f"Fold_{fold_i+1}_train_classification_losses.pkl"
        with open(variable_save_path, 'wb') as f:
            pickle.dump(train_classification_losses, f)
        
        variable_save_path = target_dir_path/f"Fold_{fold_i+1}_validation_classification_losses.pkl"
        with open(variable_save_path, 'wb') as f:
            pickle.dump(validation_classification_losses, f)
        
        variable_save_path = target_dir_path/f"Fold_{fold_i+1}_kfold_validation_classification_losses.pkl"
        with open(variable_save_path, 'wb') as f:
            pickle.dump(kfold_validation_classification_losses, f)
            
        #Saving Reconstruction Losses
        variable_save_path = target_dir_path/f"Fold_{fold_i+1}_training_reconstruction_losses.pkl"
        with open(variable_save_path, 'wb') as f:
            pickle.dump(training_reconstruction_losses, f)
        
        variable_save_path = target_dir_path/f"Fold_{fold_i+1}_train_reconstruction_losses.pkl"
        with open(variable_save_path, 'wb') as f:
            pickle.dump(train_reconstruction_losses, f)
        
        variable_save_path = target_dir_path/f"Fold_{fold_i+1}_validation_reconstruction_losses.pkl"
        with open(variable_save_path, 'wb') as f:
            pickle.dump(validation_reconstruction_losses, f)
               
timeSeriesLength = 26
timeSeriesSkip = 13
rois = 392
excluding = 78
random_states = [30, 60, 90, 120, 150]
conn_est = ConnectivityMeasure(kind = 'correlation')
noise_seed = 0

#X, X_test, y, y_test = train_test_split(abide_loaded, y_loaded, test_size=0.1, random_state=random_state)

desc = f"Component_CNN_Autoencoder_model_binaries"

kernel_sizes = [400]
lambdaL2 = [0]
learning_rates = [0.0005]
dropouts = [0.3]
noise_mean = [0]
noise_stddev = [0]
batch_sizes = [150]
alphas = [0]
for kernel_size in kernel_sizes:
    for l2 in lambdaL2:
        for learning_rate in learning_rates:
            for noise_m in noise_mean:
                for noise_sd in noise_stddev:
                    for random_state in random_states:
                        for batch_size in batch_sizes:
                            for alpha in alphas:
                                for band_pass_filtering in [True]: 
                                    for global_signal_regression in [True]:
                                        core(band_pass_filtering=band_pass_filtering, global_signal_regression=global_signal_regression, X_test=None, y_test=None, epochs=150, kernels=kernel_size, dropout=0.3, 
                                            folds=10, learning_rate=learning_rate, random_state=random_state, desc=desc, l2=l2, batch_size=batch_size, 
                                            save_model=False, mean=noise_m, stddev=noise_sd, slice_augment=True, split_augment=False, alpha=alpha, beta=1)
