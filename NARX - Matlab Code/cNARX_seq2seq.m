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
    
    %split and scale data
    teststart = datetime({'12.01.2012 00:00:00'});
    testende = datetime({'28.12.2015 00:00:00'});
    idxtraining = Data_orig.Date <teststart;
    idxtest = Data_orig.Date >= teststart & Data_orig.Date <= testende;
    teststart_idx = find(Data_orig.Date >= teststart);teststart_idx = teststart_idx(1);
    
    
    Data1 = Data_orig{:,:};
    Data = normalize(Data1,'range',[-1 1]);%scaling
    
    sequence_length = 12; %also works only for 12 in this script...
    
    Training = Data(idxtraining,:) ;
    
    %make data ready for custom neural network
    XTrain=Training(:,2:end);YTrain=Training(:,1);
    
    x1 = to_sequence_input(sequence_length,XTrain(:,1),size(XTrain,1));
    x2 = to_sequence_input(sequence_length,XTrain(:,2),size(XTrain,1));
    x3 = to_sequence_input(sequence_length,XTrain(:,3),size(XTrain,1));
    x4 = to_sequence_input(sequence_length,XTrain(:,4),size(XTrain,1));
    
    x = [x1;x2;x3;x4];
    y = to_sequence_target(sequence_length,YTrain,size(YTrain,1));
    
    
    
    %% Bayes Optimization
    %define variables which should be optimized
    optimVars = [
        optimizableVariable('hiddenLayerSize', [1,20], 'Type', 'integer')
        optimizableVariable('InputrH',  [0,1], 'Type', 'integer')
        optimizableVariable('InputT', [0,1], 'Type', 'integer')
        optimizableVariable('InputTsin', [0,1], 'Type', 'integer')
        optimizableVariable('ID_P', [1,12],'Type', 'integer')
        optimizableVariable('ID_rH', [1,12],'Type', 'integer')
        optimizableVariable('ID_T', [1,12],'Type', 'integer')
        optimizableVariable('ID_Tsin', [1,12],'Type', 'integer')
        optimizableVariable('FDmax', [1,12],'Type', 'integer')];
    
    %define function which is to minimize
    minfn = @(X)wrapNARXNet(x,y,'hiddenLayerSize',X.hiddenLayerSize,'architecture','closed',...
        'ID_P',X.ID_P,'ID_rH',X.ID_rH,'ID_T',X.ID_T,'ID_Tsin',X.ID_Tsin,...
        'FDmax',X.FDmax,'InputrH',X.InputrH,'InputT',X.InputT,'InputTsin',X.InputTsin);
    
    rng(0) %define random number seed for reproducability
    %perform optimisation:
    results = bayesopt(minfn,optimVars,'IsObjectiveDeterministic',false,...
        'AcquisitionFunctionName','expected-improvement',...%-per-second-plus',...
        'ConditionalVariableFcn',@condvariablefcn,...
        'OutputFcn',@customOutputFun,...
        'MaxObjectiveEvaluations',150);
    
    
    %% Test on independent dataset
    [bestKonfig,CriterionValue,iteration] = bestPoint(results,'Criterion','min-observed');
    
    hiddensz = bestKonfig.hiddenLayerSize;
    architecture = 'closed';%string(bestKonfig.architecture);
    feedbackDelays = 1:bestKonfig.FDmax;
    optimizer = 'trainlm';
    h = bestKonfig.hiddenLayerSize;
    
    inputs = [1 bestKonfig.InputrH bestKonfig.InputT bestKonfig.InputTsin];
    numInputs = sum(inputs);
    inimax=10;
    pp=1;
    
    %%
    TrainingData = Data_orig;
    firststart_forecast1 = datetime({'01.01.2012 00:00:00'});
    idx_forecaststart = find(TrainingData.Date >= firststart_forecast1);idx_forecaststart = idx_forecaststart(1);
    laststart_forecast1 = TrainingData.Date(end);laststart_forecast1.Day = 01;
    laststart_forecast1.Month = laststart_forecast1.Month-2;
    
    %% build/train the Model
    runmax = (13-month(firststart_forecast1))+12*(year(laststart_forecast1)-year(firststart_forecast1)-1)+month(laststart_forecast1);
    
    folderpath = strcat('.\trainednets\',Well_ID,'\');
    if ~exist(folderpath, 'dir')
        mkdir(folderpath)
    end
    c1 = clock;
    for run = 1:runmax
        
        startdate = firststart_forecast1;
        startdate.Month = startdate.Month+run-1;
        enddate = startdate;
        enddate.Month = enddate.Month+3;
        
        Training_endindex = find(TrainingData.Date < startdate);Training_endindex = Training_endindex(end);
        Testing_endindex = find(TrainingData.Date < enddate);Testing_endindex = Testing_endindex(end);
        
        
        Testing_endindex = Training_endindex+sequence_length;
        %% Calculations
        Dmax = nanmax(bestKonfig{pp,5:9})*sequence_length;
        TrainingData_n = normalize(TrainingData,'range',[-1,1]); %normalize ANN Input data
        date = TrainingData.Date;
        
        %%
        
        for ini = 1:inimax %initialization ensemble
            netfilename = string(strcat(Well_ID,"_trainedNARXNET_ini_",string(ini)));
            netfilepath = strcat(folderpath,netfilename,'.mat');
            if isfile(netfilepath)
                load(netfilepath,'net')
            else
                %% Training
                % build NARX
                rng(ini) %set random number generator for reproducability
                net = narxnet(1,feedbackDelays,h,architecture,optimizer); % Create a Nonlinear Autoregressive Network with External Input
                net.divideFcn = 'divideblock';  % Divide data blockwise (good for time series)
                net.divideMode = 'time';  % for dynamic networks (like NARX)
                net.divideParam.trainRatio = 80/100; %how much data for training
                net.divideParam.valRatio = 20/100; %how much data for early stopping
                net.divideParam.testRatio = 0;
                net.trainParam.showWindow = 0;
                net.trainParam.epochs = 30;
                net.trainParam.max_fail = 5; %early stopping
                net.numInputs = numInputs;
                net.inputConnect = [ones(1,numInputs);zeros(1,numInputs)];
                
                net.inputWeights{1,1}.delays =  1:bestKonfig.ID_P(pp);
                %                 net.inputWeights{1,1}.delays = 1:ceil(length(net.inputWeights{1,1}.delays)/sequence_length);
                net.inputs{1,1}.name = 'P';
                idx = 2;
                if inputs(2) == 1
                    net.inputWeights{1,idx}.delays =  1:bestKonfig.ID_rH(pp);
                    %                     net.inputWeights{1,idx}.delays = 1:ceil(length(net.inputWeights{1,idx}.delays)/sequence_length);
                    net.inputs{idx,1}.name = 'rH';idx = idx+1;
                end
                if inputs(3) == 1
                    net.inputWeights{1,idx}.delays =  1:bestKonfig.ID_T(pp);
                    %                     net.inputWeights{1,idx}.delays = 1:ceil(length(net.inputWeights{1,idx}.delays)/sequence_length);
                    net.inputs{idx,1}.name = 'T';idx = idx+1;
                end
                if inputs(4) == 1
                    net.inputWeights{1,idx}.delays =  1:bestKonfig.ID_Tsin(pp);
                    %                     net.inputWeights{1,idx}.delays = 1:ceil(length(net.inputWeights{1,idx}.delays)/sequence_length);
                    net.inputs{idx,1}.name = 'Tsin';
                end
                
                clear TrainingTarget
                TrainingTarget = to_sequence_target(sequence_length,TrainingData_n.(1),Training_endindex);
                x1 = to_sequence_input(sequence_length,TrainingData_n.(2),Training_endindex);
                x2 = to_sequence_input(sequence_length,TrainingData_n.(3),Training_endindex);
                x3 = to_sequence_input(sequence_length,TrainingData_n.(4),Training_endindex);
                x4 = to_sequence_input(sequence_length,TrainingData_n.(5),Training_endindex);

                TrainingInput = [x1;x2;x3;x4];
                TrainingInput = TrainingInput(logical(inputs),:);
               
                %fill delays and adapt input and target data
                [TrainingInput,ID_ini,FD_ini,TrainingTarget] = preparets(net,TrainingInput,{},TrainingTarget); %prepare data
                [net,~] = train(net,TrainingInput,TrainingTarget,ID_ini,FD_ini);%training
                
                
                save(netfilepath,'net')
                
            end
            
            %% Testing
            TestingData_n = TrainingData_n(Training_endindex+1:Testing_endindex,:);
            
            %take last part of training data
            tempInput1 = TrainingData_n{Training_endindex-Dmax+1:Training_endindex,2:end}; %needed to fill the delays prior to testing
            tempTarget1 = TrainingData_n{Training_endindex-Dmax+1:Training_endindex,1}; %needed to fill the delays prior to testing
            
            %take real testing data and attach it
            tempInput2 = TestingData_n{1:end,2:end};
            tempTarget2 = TestingData_n{:,1};
            
            TestingInput = [tempInput1;tempInput2];
            TestingTarget = [tempTarget1;tempTarget2];
            
            endidx = size(TestingTarget,1);
            TestingTarget = to_sequence_target(sequence_length,TestingTarget(:,1),endidx);
            xt1 = to_sequence_input(sequence_length,TestingInput(:,1),endidx);
            xt2 = to_sequence_input(sequence_length,TestingInput(:,2),endidx);
            xt3 = to_sequence_input(sequence_length,TestingInput(:,3),endidx);
            xt4 = to_sequence_input(sequence_length,TestingInput(:,4),endidx);
            
            
            %NNData
            TestingInput = [xt1;xt2;xt3;xt4];
            TestingInput = TestingInput(logical(inputs),:);

            
            
            %remove values not needed
            delays_needed = max(bestKonfig{pp,5:9});
            TestingInput(:,1:end-delays_needed-1)=[];
            TestingTarget(:,1:end-delays_needed-1)=[];
            
            net = closeloop(net);  %testing is performed in closed loop always
            [TestingInput,ID_ini,FD_ini,~,~,~] = preparets(net,TestingInput,{},TestingTarget);%prepare data
            
            TestOutput = net(TestingInput,ID_ini,FD_ini); % simulate prediction
            
            TestResults1 = table;
            TestResults1.Date =  TestingData_n.Date;
            for it = 1:size(TestOutput,2)
                TestResults1{it:it+sequence_length-1,1+it} = cell2mat(TestOutput(it))';
            end
            TestResults1 = table2timetable(TestResults1);
            
            %Redo Normalization
            for i = 1:size(TestResults1,2)
                TestResults1{:,i}=(((TestResults1{:,i}+1)/2)*(max(TrainingData.GWL)-min(TrainingData.GWL)))+min(TrainingData.GWL);
            end
            
            %Summarize TestResults
            if run == 1 && ini == 1; TestResults = TestResults1;
            else, TestResults = synchronize(TestResults,TestResults1);    end
            
            
        end
    end
    c2 = clock;
    %% tidy up
    
    for r = 1:size(TestResults,2)
        TestResults.Properties.VariableNames{r} = strcat('V',num2str(r));
    end
    
    %% Evaluation
    
    TestResults_median = table;
    TestResults_median.Date = TestResults.Date;
    for run = 1:runmax
        TestResults_median{:,run+1} = median(TestResults{:,run*(inimax)-(inimax-1):run*(inimax)},2);
    end
    TestResults_median = table2timetable(TestResults_median);
    
    Forecast_Errors = NaN(8,runmax);
    for run = 1:runmax
        
        temp1=TestResults_median(:,run);temp1=rmmissing(temp1);
        temp =synchronize(GWData_int,temp1,'intersection');
        
        obs = temp{:,1};
        sim = temp{:,2};
        
        ts = find(GWData_int.Date == temp.Date(1));
        te = find(GWData_int.Date == temp.Date(end));
        obsPI12 = GWData_int.GWL(ts-12:te-12);
        obsPI12op = GWData_int.GWL(ts-1);
        err = gsubtract(sim,obs); %error
        err_rel = gsubtract(sim,obs)./range(TrainingData.GWL);%relative error
        err_nash = gsubtract(obs,nanmean(TrainingData.GWL));
        err_PI12 = gsubtract(obs,obsPI12);
        err_PI12op = obs-obsPI12op;
        %------
        Forecast_Errors(1,run) = 1-(nansum(err.^2)/nansum(err_nash.^2));             % [ ] Nash-Sutcliffe-Efficiency
        Forecast_Errors(2,run) = corr(obs,sim,'rows','complete')^2;                  % [ ] Pearson-Corr. squared
        Forecast_Errors(3,run) = sqrt(nanmean((err).^2));                            % [m] RMSE, Root Mean Squared Error
        Forecast_Errors(4,run) = sqrt(nanmean((err_rel).^2))*100;                    % [%] RMSEr, Relative Root Mean Squared Error
        Forecast_Errors(5,run) = nanmean(err);                                       % [m] BIAS, systematic error
        Forecast_Errors(6,run) = nanmean(err_rel)*100;                               % [%] Relative BIAS
        Forecast_Errors(7,run) = 1-(nansum(err.^2)/nansum(err_PI12.^2));             % [ ] PI12, Persistency Index 12 week / 3 month forecast - compares with values 12 steps prior
        Forecast_Errors(8,run) = 1-(nansum(err.^2)/nansum(err_PI12op.^2));           % [ ] PI12op, Persistency Index 12 week / 3 month forecast - compares with las observes value
        
        
    end
    time2 = datetime('now');
    
    scores = median(Forecast_Errors,2);
    
    %% Visualisation
    
    figure('Position',[235,255,1409,563.6]);
    s = subplot(1,1,1);
    
    plot(TrainingData.Date(idx_forecaststart:end),TrainingData.GWL(idx_forecaststart:end),'b-','LineWidth',1.5), grid off, hold on
    yl = ylim;
    
    plot(TestResults_median.Date,TestResults_median.(1),'r-','LineWidth',1.5,'HandleVisibility','on')
    for i = 2:size(TestResults_median,2)
        plot(TestResults_median.Date,TestResults_median.(i),'r-','LineWidth',1.5,'HandleVisibility','off')
    end
    
    for i = 1:size(TestResults,2)
        plot(TestResults.Date,TestResults.(i),'Color',[1 0 0 0.1],'LineWidth',1.2,'HandleVisibility','off')
    end
    
    title(strcat("NARX - Forecast 3 Months seq2seq: ",strrep(Well_ID,'_','\_')),'FontSize',12,'FontWeight','normal');
    grid off
    xlabel('Date')
    ylabel('GWL [m asl]')
    %     leg = legend('obs','sim','Location','northeastoutside');
    s.Position(3) = s.Position(3)*0.91;
    legend('observed','simulated median','Position',[0.8615,0.8608,0.12,0.0647]);
    datetick('x','yyyy-mm','keepticks')
    
    %Error Box
    %     elapsedtime = time2-time1;
    
    dim = [0.8615,0.25,0.10,0.59];
    %str1 = string(bestKonfig.Properties.VariableNames(2:10));str1 = strrep(str1,'_',' ');
    str1 = ["HiddenSize","rH","T","Tsin","ID P","ID rH","ID T","ID Tsin","FD"];
    str2 = string(bestKonfig{:,:});
    str2(ismissing(str2))="-";str2(2:4) = strrep(str2(2:4),'0','No');str2(2:4) = strrep(str2(2:4),'1','Yes');
    str1_2 = {["NSE [ ]" ],["R² [ ]"],["RMSE [m]" ],["rRMSE [%]"],["Bias [m]" ],["rBias [%]"],["PI12 [ ]"],["PI12op [ ]"],[" "]};
    str2_2 = {sprintf('%0.2f\n%0.2f\n%0.2f\n%0.2f\n%0.2f\n%0.2f\n%0.2f\n%0.2f\n',...
        scores)};
    annotation('textbox',dim,'String',[str1_2,str1]);
    annotation('textbox',dim+[0.06,0,0,0],'String',[str2_2,str2],'EdgeColor','none');
    
    print('-dpng','-r300',strcat(".\Forecast_NARX_seq2seq_Paper_",Well_ID))
    
    %% save logs
    fileID = fopen('log_summary_seq2seq_NARX_'+Well_ID+'.txt', 'w');
    fprintf(fileID,...
        "\nBEST:\n\nNSE = %.2f\nR²  = %.2f\nRMSE = %.2f\nrRMSE = %.2f\nBias = %.2f\nrBias = %.2f\nPI12 = %.2f\nPI12op = %.2f\n\n",...
        scores);
    
    fprintf(fileID,...
        "hiddensize: %d\nrH = %d\nT = %d\nTsin = %d\nID P = %d\nID rH = %d\nID T = %d\nID Tsin = %d\nFDmax = %d\n\n\n",...
        bestKonfig{:,:});
    
    for i = 1:size(results.XTrace,1)
        fprintf(fileID,...
            "Iteration %d:  {'target': %f, 'params': {'hiddensize': %d, 'rH': %d, 'T': %d, 'Tsin': %d, 'ID P': %d, 'ID rH': %d, 'ID T': %d, 'ID Tsin': %d, 'FDmax': %d}}\n",...
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
    toc
    save(strcat("Workspace_Bayesopt_seq2seq_",Well_ID));
end

cd('C:\Users\Andreas Wunsch\Workspace\01_Matlab')
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
ip.addParameter('FDmax', 1);
ip.addParameter('architecture', 1);
ip.addParameter('InputrH', 1);
ip.addParameter('InputT', 1);
ip.addParameter('InputTsin', 1);
% ip.addParameter('TestTarget', 1);
% ip.addParameter('TestInput', 1);
parse(ip, varargin{:});

hiddensz = ip.Results.hiddenLayerSize;
architecture = string(ip.Results.architecture);
feedbackDelays = 1:ip.Results.FDmax;
optimizer = 'trainlm';

inputs = [1 ip.Results.InputrH ip.Results.InputT ip.Results.InputTsin];
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
    net.trainParam.showWindow = 1;
    net.trainParam.time = 60*5; %maximum training time is 60 seconds x 5
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
        if size(results.ObjectiveTrace,1) > 50 %at least 50 iterations before termination criterion
            minpos = find(results.ObjectiveTrace == min(results.ObjectiveTrace));
            if size(results.ObjectiveTrace,1)-minpos > 20 %if after 20 iterations no improvement, then abort
                stop = true;
            end
        end
    case 'done'
        stop = true;
        
end
end

function [x1] = to_sequence_input(sequence_length,Data_n,endindex)
    for seq = 1:sequence_length
        Input1(:,seq) = Data_n(0+seq:endindex-2*sequence_length+seq,1);
    end
    i=1;
    x1_1 = num2cell(Input1(:,i)');i=i+1;
    x1_2 = num2cell(Input1(:,i)');i=i+1;
    x1_3 = num2cell(Input1(:,i)');i=i+1;
    x1_4 = num2cell(Input1(:,i)');i=i+1;
    x1_5 = num2cell(Input1(:,i)');i=i+1;
    x1_6 = num2cell(Input1(:,i)');i=i+1;
    x1_7 = num2cell(Input1(:,i)');i=i+1;
    x1_8 = num2cell(Input1(:,i)');i=i+1;
    x1_9 = num2cell(Input1(:,i)');i=i+1;
    x1_10 = num2cell(Input1(:,i)');i=i+1;
    x1_11= num2cell(Input1(:,i)');i=i+1;
    x1_12 = num2cell(Input1(:,i)');i=i+1;
    x1 = catsamples(x1_1,x1_2,x1_3,x1_4,x1_5,x1_6,x1_7,x1_8,x1_9,x1_10,x1_11,x1_12,'pad');
end

function [sequence] = to_sequence_target(sequence_length,Data_n,endindex)
    for seq = 1:sequence_length
        Target(:,seq) = Data_n(sequence_length+seq:endindex-sequence_length+seq,1);
    end
    i=1;
    t1 = num2cell(Target(:,i)');i=i+1;
    t2 = num2cell(Target(:,i)');i=i+1;
    t3 = num2cell(Target(:,i)');i=i+1;
    t4 = num2cell(Target(:,i)');i=i+1;
    t5 = num2cell(Target(:,i)');i=i+1;
    t6 = num2cell(Target(:,i)');i=i+1;
    t7 = num2cell(Target(:,i)');i=i+1;
    t8 = num2cell(Target(:,i)');i=i+1;
    t9 = num2cell(Target(:,i)');i=i+1;
    t10 = num2cell(Target(:,i)');i=i+1;
    t11= num2cell(Target(:,i)');i=i+1;
    t12 = num2cell(Target(:,i)');i=i+1;
    sequence = catsamples(t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,'pad');
end