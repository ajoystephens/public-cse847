clear, clc;

% ADD SLEP FUNCTIONS TO PATH
root=cd;
addpath(genpath([root '/SLEP']));

% GET DATA
filename = 'data/alzheimers/ad_data.mat';
data = load(filename);


% LOOP THROUGH POTENTIAL L1 PARAMETERS
for par = [0.001, 0.005, 0.999, 0.01, 0.25, 0.05, ...
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    disp(['--- par: ',num2str(par),' -----------------------------------']);

    % TRAIN MODEL
    [w, c] = logistic_l1_train(data.X_train, data.y_train, par);

    % TRAIN AUC
    pred_train = c+(data.X_test*w);
    pred_train( pred_train <= 0 ) = -1;
    pred_train( pred_train > 0 ) = 1;
    [X,Y,T,AUC] = perfcurve(data.y_test,pred_train,1);
    disp(['TRAIN AUC: ',num2str(AUC)]);

    % TEST AUC
    pred_test = c+(data.X_test*w);
    pred_test( pred_test <= 0 ) = -1;
    pred_test( pred_test > 0 ) = 1;
    [X,Y,T,AUC] = perfcurve(data.y_test,pred_test,1);
    disp(['TEST AUC: ',num2str(AUC)]);

end
