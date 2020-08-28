% written by: Andreas Wunsch, andreas.wunsch@kit.edu
% year: 2020
% ORCID: https://orcid.org/0000-0002-0585-9549
%%
clear
clc
close all
tic

%% Define IDs of Wells, which are to be optimized

% % e.g.
% load Well_IDs.mat
Well_IDs = "this_is_an_ID";

%% Loop over all Wells (w)

for w = 1:size(Well_IDs,1)
    Well_ID = Well_IDs(w);
    disp(Well_ID)
    
    %% load Data
    % GW-Data should be named: "GWData_int", type: timetable, Variablenames: "Date","GWL"; 
    % Meteorological Date should be named: "HYRASdata", type: timetable, Variablenames: "Date","P","rH","T","Tsin"; 
    
%     load(strcat(Well_ID,'_weeklyData_HYRAS')); 
%     load(strcat(Well_ID,'_GW-Data')); 
    load exampledata.mat
    
    %% some data stuff
    Data_orig = synchronize(GWData_int,HYRASdata,'intersection'); %combine Data
    %add shifted GWL
    GWL_shift = GWData_int;
    GWL_shift.Date = GWL_shift.Date + 7;
    GWL_shift.Properties.VariableNames = {'GWLt-1'};
    Data_orig = synchronize(Data_orig,GWL_shift,'intersection'); %combine Data
    
    %split and scale data
    teststart = datetime({'12.01.2012 00:00:00'}); %define start of test period
    testend = datetime({'28.12.2015 00:00:00'}); %define end of test period
    
    idxtraining = Data_orig.Date <teststart;
    idxtest = Data_orig.Date >= teststart & Data_orig.Date <= testend;
    
    Data1 = Data_orig{:,:};
    Data = normalize(Data1,'range',[-1 1]);%scaling
    
    Training = Data(idxtraining,:) ;
    Testing = Data(idxtest,:);
    
    %make data ready for custom neural network
    XTrain=Training(:,2:end);YTrain=Training(:,1);
    XTest=Testing(:,2:end);YTest=Testing(:,1);
    x1=tonndata(XTrain(:,1),false,false);
    x2=tonndata(XTrain(:,2),false,false);
    x3=tonndata(XTrain(:,3),false,false);
    x4=tonndata(XTrain(:,4),false,false);
    x5=tonndata(XTrain(:,5),false,false);
    x = [x1;x2;x3;x4;x5];
    y=tonndata(YTrain,false,false);
    x1=tonndata(XTest(:,1),false,false);
    x2=tonndata(XTest(:,2),false,false);
    x3=tonndata(XTest(:,3),false,false);
    x4=tonndata(XTest(:,4),false,false);
    x5=tonndata(XTest(:,5),false,false);
    xt = [x1;x2;x3;x4;x5]; clear x1 x2 x3 x4 x5 P
    yt=tonndata(YTest,false,false);
    
    %% Bayes Optimization
    %define variables which should be optimized
    optimVars = [
        optimizableVariable('hiddenLayerSize', [1,20], 'Type', 'integer')
        optimizableVariable('InputrH',  [0,1], 'Type', 'integer')
        optimizableVariable('InputT', [0,1], 'Type', 'integer')
        optimizableVariable('InputTsin', [0,1], 'Type', 'integer')
        optimizableVariable('ID_P', [1,52],'Type', 'integer')
        optimizableVariable('ID_rH', [1,52],'Type', 'integer')
        optimizableVariable('ID_T', [1,52],'Type', 'integer')
        optimizableVariable('ID_Tsin', [1,52],'Type', 'integer')
        optimizableVariable('ID_GWLshift', [1,52],'Type', 'integer')
        optimizableVariable('FDmax', [1,52],'Type', 'integer')];
    
    %define function which is to minimize
    minfn = @(X)wrapNARXNet(x,y,'hiddenLayerSize',X.hiddenLayerSize,'architecture','closed',...
        'ID_P',X.ID_P,'ID_rH',X.ID_rH,'ID_T',X.ID_T,'ID_Tsin',X.ID_Tsin,'ID_GWLshift',X.ID_GWLshift,...
        'FDmax',X.FDmax,'InputrH',X.InputrH,'InputT',X.InputT,'InputTsin',X.InputTsin,...
        'TestTarget',yt,'TestInput',xt);
    
    rng(0) %define random number seed for reproducability
    %perform optimisation:
    results = bayesopt(minfn,optimVars,'IsObjectiveDeterministic',false,...
        'AcquisitionFunctionName','expected-improvement',...
        'ConditionalVariableFcn',@condvariablefcn,...
        'OutputFcn',@customOutputFun,...
        'MaxObjectiveEvaluations',50);
    
    
    %% Test on independent dataset
    [bestKonfig,CriterionValue,iteration] = bestPoint(results,'Criterion','min-observed');
    
    hiddensz = bestKonfig.hiddenLayerSize;
    architecture = 'closed'; %open loop does not work for optimization
    feedbackDelays = 1:bestKonfig.FDmax;
    optimizer = 'trainlm';
    
    inputs = [1 bestKonfig.InputrH bestKonfig.InputT bestKonfig.InputTsin 1];
    numInputs = sum(inputs);
    x1 = x(logical(inputs),:);
    xt1 = xt(logical(inputs),:);
    
    inimax = 10;
    c1 = clock;
    for ini = 1:inimax
        rng(ini-1)
        net = narxnet(1,feedbackDelays,hiddensz,architecture,optimizer); % Create a Nonlinear Autoregressive Network with External Input
        net.divideFcn = 'divideblock';  % Divide data blockwise (good for time series)
        net.divideMode = 'time';  % for dynamic networks (like NARX)
        net.divideParam.trainRatio = 80/100; %how much data for training
        net.divideParam.valRatio = 20/100; %how much data for early stopping
        net.divideParam.testRatio = 0;
        net.trainParam.showWindow = 0;
        %net.trainParam.time = 60; %maximum training time is 60 seconds
        net.trainParam.epochs = 30;
        net.trainParam.max_fail = 5; %early stopping
        %-------------------
        
        net.numInputs = numInputs;
        net.inputConnect = [ones(1,numInputs);zeros(1,numInputs)];
        
        net.inputWeights{1,1}.delays =  1:bestKonfig.ID_P;
        net.inputs{1,1}.name = 'P';
        idx = 2;
        if inputs(2) == 1
            net.inputWeights{1,idx}.delays =  1:bestKonfig.ID_rH;
            net.inputs{idx,1}.name = 'rH';idx = idx+1;
        end
        if inputs(3) == 1
            net.inputWeights{1,idx}.delays =  1:bestKonfig.ID_T;
            net.inputs{idx,1}.name = 'T';idx = idx+1;
        end
        if inputs(4) == 1
            net.inputWeights{1,idx}.delays =  1:bestKonfig.ID_Tsin;
            net.inputs{idx,1}.name = 'Tsin';idx = idx+1;
        end
        if inputs(5) == 1
            net.inputWeights{1,idx}.delays =  1:bestKonfig.ID_GWLshift;
            net.inputs{idx,1}.name = 'GWLt-1';idx = idx+1;
        end
        
        
        net.outputs{1,2}.name = 'GWL';
        % view(net) %show custom NN
        
        [TrainingInput,ID_ini,FD_ini,TrainingTarget] = preparets(net,x1,{},y); %prepare data
        nets = train(net,TrainingInput,TrainingTarget,ID_ini,FD_ini);%training
        [ypred,xf,af] = nets(TrainingInput,ID_ini,FD_ini); %apply
        
        [nets,xf,af] = closeloop(nets,xf,af); %close loop if not closed yet
        testpred(ini,:) = nets(xt1,xf,af); %apply for prediction
    end
    c2 = clock;
    %% rescale
    sim =(((cell2mat(testpred)'+1)/2)*(max(Data1(:,1))-min(Data1(:,1))))+min(Data1(:,1));
    obs = (((cell2mat(yt)'+1)/2)*(max(Data1(:,1))-min(Data1(:,1))))+min(Data1(:,1));
    sim_median = median(sim,2);
    obsPI1 = Data1(idxtest,6);
    %% error measures
    err = gsubtract(sim_median,obs); %error
    err_rel = gsubtract(sim_median,obs)./range(Data1(:,1));%relative error
    err_nash = gsubtract(obs,nanmean(Data1(idxtraining,1)));
    err_PI = gsubtract(obs,obsPI1);
    
    mse = mean((err).^2);
    nse = 1-(nansum(err.^2)/nansum(err_nash.^2));         % [ ] Nash-Sutcliffe-Efficiency
    r2 = corr(obs,sim_median,'rows','complete')^2;        % [ ] Pearson-Corr. squared
    rmse = sqrt(nanmean((err).^2));                       % [m] RMSE, Root Mean Squared Error
    rmse_n = sqrt(nanmean((err_rel).^2))*100;             % [%] RMSEr, Relative Root Mean Squared Error
    bias = nanmean(err);                                  % [m] BIAS, systematic error
    bias_n = nanmean(err_rel)*100;                        % [%] Relative BIAS
    PI = 1-(nansum(err.^2)/nansum(err_PI.^2));            % [ ] Persistency Index
    
    %calculate ensemble member errors
    errors = table;
    for i = 1:inimax
        sim_temp = sim(:,i);
        err = gsubtract(sim_temp,obs); %error
        err_rel = gsubtract(sim_temp,obs)./range(Data1(:,1));%relative error
        mse = mean((err).^2);
        errors.NSE(i) = 1-(nansum(err.^2)/nansum(err_nash.^2));       % [ ] Nash-Sutcliffe-Efficiency
        errors.R2(i) = corr(obs,sim_temp,'rows','complete')^2;        % [ ] Pearson-Corr. squared
        errors.RMSE(i) = sqrt(nanmean((err).^2));                     % [m] RMSE, Root Mean Squared Error
        errors.rRMSE(i) = sqrt(nanmean((err_rel).^2))*100;            % [%] RMSEr, Relative Root Mean Squared Error
        errors.Bias(i) = nanmean(err);                                % [m] BIAS, systematic error
        errors.rBias(i) = nanmean(err_rel)*100;                       % [%] Relative BIAS
        errors.PI(i) = 1-(nansum(err.^2)/nansum(err_PI.^2));          % [ ] Persistency Index
    end
    writetable(errors,strcat('ensemble_member_errors_NARX_',Well_ID,'.txt'),'WriteVariableNames',true,'Delimiter',';')
    
    %% plot
    figure('Position',[235,255,1409,563.6]);
    s = subplot(1,1,1);
    plot(Data_orig.Date(idxtest),obs,'b','LineWidth',1.5);hold on
    
    for i = 1:size(sim,2)
        plot(Data_orig.Date(idxtest),sim(:,i),'Color',[1 0 0 0.1],'HandleVisibility','off','LineWidth',1.5)
    end
    plot(Data_orig.Date(idxtest),sim_median,'r','LineWidth',1.5),hold on,
    title(strcat("NARX: ",strrep(Well_ID,'_','\_')),'FontSize',12,'FontWeight','normal');
    grid off
    xlabel('Date')
    ylabel('GWL [m asl]')
    s.Position(3) = s.Position(3)*0.91;
    legend('observed','simulated median','Position',[0.8615,0.8608,0.12,0.0647]);
    datetick('x','yyyy-mm','keepticks')
        
    dim = [0.8615,0.25,0.10,0.58];
    str1 = ["HiddenSize","rH","T","Tsin","ID P","ID rH","ID T","ID Tsin","ID GWLt-1","FD"];
    str2 = string(bestKonfig{:,:});
    str2(ismissing(str2))="-";str2(2:4) = strrep(str2(2:4),'0','No');str2(2:4) = strrep(str2(2:4),'1','Yes');
    str1_2 = {["NSE [ ]" ],["R² [ ]"],["RMSE [m]" ],["rRMSE [%]"],["Bias [m]" ],["rBias [%]"],["PI [ ]"],[" "]};
    str2_2 = {sprintf('%0.2f\n%0.2f\n%0.2f\n%0.2f\n%0.2f\n%0.2f\n%0.2f\n',...
        nse,r2,rmse,rmse_n,bias,bias_n,PI)};
    annotation('textbox',dim,'String',[str1_2,str1]);
    annotation('textbox',dim+[0.06,0,0,0],'String',[str2_2,str2],'EdgeColor','none');
    
    print('-dpng','-r300',strcat("Test_Bayesopt_bestKonfig_",Well_ID))
    
    %% save logs
    fileID = fopen('log_summary_NARX_'+Well_ID+'.txt', 'w');
    fprintf(fileID,...
        "\nBEST:\n\nNSE = %.2f\nR²  = %.2f\nRMSE = %.2f\nrRMSE = %.2f\nBias = %.2f\nrBias = %.2f\nPI = %.2f\n\n",...
        nse,r2,rmse,rmse_n,bias,bias_n,PI...
        );
    
    fprintf(fileID,...
        "hiddensize: %d\nrH = %d\nT = %d\nTsin = %d\nID P = %d\nID rH = %d\nID T = %d\nID Tsin = %d\nID GWLt-1 = %d\nFDmax = %d\n\n\n",...
        bestKonfig{:,:});
    
    for i = 1:size(results.XTrace,1)
        fprintf(fileID,...
            "Iteration %d:  {'target': %f, 'params': {'hiddensize': %d, 'rH': %d, 'T': %d, 'Tsin': %d, 'ID P': %d, 'ID rH': %d, 'ID T': %d, 'ID Tsin': %d, 'ID GWLt-1': %d, 'FDmax': %d}}\n",...
            i,-1*results.ObjectiveTrace(i),results.XTrace{i,:});
    end
    fclose(fileID);
    
    %timelog
    fileID = fopen('timelog_NARX_'+Well_ID+'.txt', 'w');
    fprintf(fileID,'Time [s] for Test-Eval (10 inis)\n');
    fprintf(fileID,"%.2f\n\n",etime(c2,c1));    
    fprintf(fileID,'Seconds per Iteration\n');
    for i = 1:size(results.IterationTimeTrace,1)
        fprintf(fileID,"%.2f\n",...
            results.IterationTimeTrace(i));
    end
    fclose(fileID);
    
    
    %% save workspace
%     toc
    save(strcat("Workspace_Bayesopt_varInputs_",Well_ID));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% functions
function target = wrapNARXNet(x, y, varargin)
% Handle variable inputs
ip = inputParser;
ip.addParameter('hiddenLayerSize', 1);
ip.addParameter('ID_P', 1);
ip.addParameter('ID_rH', 1);
ip.addParameter('ID_T', 1);
ip.addParameter('ID_Tsin', 1);
ip.addParameter('ID_GWLshift', 1);
ip.addParameter('FDmax', 1);
ip.addParameter('architecture', 1);
ip.addParameter('InputrH', 1);
ip.addParameter('InputT', 1);
ip.addParameter('InputTsin', 1);
ip.addParameter('TestTarget', 1);
ip.addParameter('TestInput', 1);
parse(ip, varargin{:});

hiddensz = ip.Results.hiddenLayerSize;
architecture = string(ip.Results.architecture);
feedbackDelays = 1:ip.Results.FDmax;
optimizer = 'trainlm';

inputs = [1 ip.Results.InputrH ip.Results.InputT ip.Results.InputTsin 1];
numInputs = sum(inputs);
x = x(logical(inputs),:);
inimax = 5;
target = NaN(inimax,1);
for ini = 1:inimax %perform several runs due to dependency in random seed(ini)
    % rng(0)
    rng(ini)
    net = narxnet(1,feedbackDelays,hiddensz,architecture,optimizer); % Create a Nonlinear Autoregressive Network with External Input
    net.divideFcn = 'divideblock';  % Divide data blockwise (good for time series)
    net.divideMode = 'time';  % for dynamic networks (like NARX)
    net.divideParam.trainRatio = 80/100; %how much data for training
    net.divideParam.valRatio = 10/100; %how much data for early stopping
    net.divideParam.testRatio = 10/100;
    net.trainParam.showWindow = 0;
    net.trainParam.epochs = 30; %max epochs
    net.trainParam.max_fail = 5; %early stopping
    %-------------------
    
    net.numInputs = numInputs;
    net.inputConnect = [ones(1,numInputs);zeros(1,numInputs)];
    
    net.inputWeights{1,1}.delays =  1:ip.Results.ID_P;
    net.inputs{1,1}.name = 'P';
    idx = 2;
    if inputs(2) == 1
        net.inputWeights{1,idx}.delays =  1:ip.Results.ID_rH;
        net.inputs{idx,1}.name = 'rH';idx = idx+1;
    end
    if inputs(3) == 1
        net.inputWeights{1,idx}.delays =  1:ip.Results.ID_T;
        net.inputs{idx,1}.name = 'T';idx = idx+1;
    end
    if inputs(4) == 1
        net.inputWeights{1,idx}.delays =  1:ip.Results.ID_Tsin;
        net.inputs{idx,1}.name = 'Tsin';
    end
    
    net.outputs{1,2}.name = 'GWL';
    % view(net)
    
    [TrainingInput,ID_ini,FD_ini,TrainingTarget] = preparets(net,x,{},y); %prepare data
    nets = train(net,TrainingInput,TrainingTarget,ID_ini,FD_ini);%training
    fprintf('.') %print dot to console
    
    % Evaluate on test set and compute errors (nsemod+r²)
    [ypred,~,~] = nets(TrainingInput,ID_ini,FD_ini);
    idx = false(size(TrainingTarget));idx(round(0.9*size(TrainingTarget,2):size(TrainingTarget,2)))=true;
    % mse = mean((cell2mat(ypred(idx))-cell2mat(TrainingTarget(idx))).^2);
    nsemod = -1*(1 -(sum((cell2mat(ypred(idx))-cell2mat(TrainingTarget(idx))).^2)/sum((mean(cell2mat(y))-cell2mat(TrainingTarget(idx))).^2)));
    r2 = corr(cell2mat(TrainingTarget(idx))',cell2mat(ypred(idx))','rows','complete')^2;
    
    target(ini,1) = nsemod-r2; %this is the objective value
end
target = mean(target);
fprintf(repmat('\b', 1, inimax)); %remove dots from console
end

function Xnew = condvariablefcn(X)
Xnew = X;
for i = 1:size(X,1)
    %set Delays to NaN if respective Input was not used
    if Xnew.InputrH(i) == 0; Xnew.ID_rH(i) = NaN; end
    if Xnew.InputT(i) == 0; Xnew.ID_T(i) = NaN; end
    if Xnew.InputTsin(i) == 0; Xnew.ID_Tsin(i) = NaN; end
end
end

function stop = customOutputFun(results,state)

stop = false;
switch state
    case 'initial'
    case 'iteration'
        if size(results.ObjectiveTrace,1) > 25 %mindestens 25 iterationen bevor Abbruchkriterium
            minpos = find(results.ObjectiveTrace == min(results.ObjectiveTrace));
            if size(results.ObjectiveTrace,1)-minpos > 10 %wenn nach 10 Iterationen keine Verbesserung, dann Abbruch
                stop = true;
            end
        end
    case 'done'
        stop = true;
        
end
end

