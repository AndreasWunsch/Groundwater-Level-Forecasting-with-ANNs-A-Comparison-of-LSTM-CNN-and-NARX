% written by: Andreas Wunsch, andreas.wunsch@kit.edu
% year: 2020
% ORCID: https://orcid.org/0000-0002-0585-9549
%%
clear
clc
close all
tic

%% load best Konfig from HP-Opt. 
load bestKonfigs_GWL_shift_example.mat

for pp = 1:size(bestKonfig,1)
    
    time1 = datetime('now');
    
    %% Define Parameters and load Data
    warning('off', 'all')
    Well_ID = bestKonfig.ID(pp);
    disp(Well_ID);
    

    % GW-Data should be named: "GWData_int", type: timetable, Variablenames: "Date","GWL"; 
    % Meteorological Date should be named: "HYRASdata", type: timetable, Variablenames: "Date","P","rH","T","Tsin"; 
    
%     load(strcat(Well_ID,'_weeklyData_HYRAS')); 
%     load(strcat(Well_ID,'_GW-Data')); 
    load exampledata.mat
    
    %%      
    TrainingData = synchronize(GWData_int,HYRASdata,'intersection');
    
    %add shifted GWL
    GWL_shift = GWData_int;
    GWL_shift.Date = GWL_shift.Date + 7;
    GWL_shift.Properties.VariableNames = {'GWLt-1'};
    TrainingData = synchronize(TrainingData,GWL_shift,'intersection');
    
    firststart_forecast1 = datetime({'01.01.2012 00:00:00'});
    idx_forecaststart = find(TrainingData.Date >= firststart_forecast1);idx_forecaststart = idx_forecaststart(1);
    laststart_forecast1 = datetime({'28.12.2015 00:00:00'});
    laststart_forecast1.Month = laststart_forecast1.Month-2;
    
    prefix = "";
    trainFcn = "trainlm";  % Levenberg-Marquardt backpropagation training function
    architecture = 'closed'; % NARX architecture for training, testing is always closed
    h = bestKonfig.h(pp); % hidden layer size
    feedbackDelays = 1:bestKonfig.FDmax(pp);
    inputs = [1 bestKonfig.InputrH(pp) bestKonfig.InputT(pp) bestKonfig.InputTsin(pp) 1];
    numInputs = sum(inputs);
    inimax=10;
    
    %% build/train the Model
    runmax = (13-month(firststart_forecast1))+12*(year(laststart_forecast1)-year(firststart_forecast1)-1)+month(laststart_forecast1);
    
    folderpath = strcat('.\trainednets\',Well_ID,'\');
    if ~exist(folderpath, 'dir')
        mkdir(folderpath)
    end
    
    for run = 1:runmax
        
        startdate = firststart_forecast1;
        startdate.Month = startdate.Month+run-1;
        enddate = startdate;
        enddate.Month = enddate.Month+3;
        
        Training_endindex = find(TrainingData.Date < startdate);Training_endindex = Training_endindex(end);
        Testing_endindex = find(TrainingData.Date < enddate);Testing_endindex = Testing_endindex(end);
        
        %% Calculations
        Dmax = nanmax(bestKonfig{pp,6:11});
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
                net = narxnet(1,feedbackDelays,h,architecture,trainFcn); % Create a Nonlinear Autoregressive Network with External Input
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
                net.inputs{1,1}.name = 'P';
                idx = 2;
                if inputs(2) == 1
                    net.inputWeights{1,idx}.delays =  1:bestKonfig.ID_rH(pp);
                    net.inputs{idx,1}.name = 'rH';idx = idx+1;
                end
                if inputs(3) == 1
                    net.inputWeights{1,idx}.delays =  1:bestKonfig.ID_T(pp);
                    net.inputs{idx,1}.name = 'T';idx = idx+1;
                end
                if inputs(4) == 1
                    net.inputWeights{1,idx}.delays =  1:bestKonfig.ID_Tsin(pp);
                    net.inputs{idx,1}.name = 'Tsin';
                end
                if inputs(5) == 1
                    net.inputWeights{1,idx}.delays =  1:bestKonfig.ID_GWLshift(pp);
                    net.inputs{idx,1}.name = 'GWLshift';idx = idx+1;
                end
                
                %             view(net)
                
                %data
                TrainingInput = TrainingData_n{1:Training_endindex,2:end};
                TrainingTarget = TrainingData_n{1:Training_endindex,1};

                x1=tonndata(TrainingInput(:,1),false,false);
                x2=tonndata(TrainingInput(:,2),false,false);
                x3=tonndata(TrainingInput(:,3),false,false);
                x4=tonndata(TrainingInput(:,4),false,false);
                x5=tonndata(TrainingInput(:,5),false,false);
                TrainingInput = [x1;x2;x3;x4;x5];
                TrainingInput = TrainingInput(logical(inputs),:);
                TrainingTarget = tonndata(TrainingTarget,false,false);
                
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
            tempInput2 = TestingData_n{:,2:end};
            tempTarget2 = TestingData_n{:,1};
            
            TestingInput = [tempInput1;tempInput2];
            TestingTarget = [tempTarget1;tempTarget2];
            
            %NNData
            xt1=tonndata(TestingInput(:,1),false,false);
            xt2=tonndata(TestingInput(:,2),false,false);
            xt3=tonndata(TestingInput(:,3),false,false);
            xt4=tonndata(TestingInput(:,4),false,false);
            xt5=tonndata(TestingInput(:,5),false,false);
            TestingInput = [xt1;xt2;xt3;xt4;xt5];
            TestingInput = TestingInput(logical(inputs),:);
            TestingTarget = tonndata(TestingTarget,false,false);%prepare data format
            
            net = closeloop(net);  %testing is performed in closed loop always
            [TestingInput,ID_ini,FD_ini,~,~,~] = preparets(net,TestingInput,{},TestingTarget);%prepare data
            
            TestOutput = net(TestingInput,ID_ini,FD_ini); % simulate prediction
            
            TestResults1 = table;
            TestResults1.Date =  TestingData_n.Date;
            TestResults1.TestOutput = cell2mat(TestOutput');
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
    
    %%
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
        Forecast_Errors(7,run) = 1-(nansum(err.^2)/nansum(err_PI12.^2));             % [ ] PI12, Persistency Index 12 week / 3 month forecast
        Forecast_Errors(8,run) = 1-(nansum(err.^2)/nansum(err_PI12op.^2));           % [ ] PI12op, Persistency Index 12 week / 3 month forecast
        
        
    end
    time2 = datetime('now');
    
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
    
    title(strcat("NARX - Forecast 3 Months seq2val: ",strrep(Well_ID,'_','\_')),'FontSize',12,'FontWeight','normal');
    grid off
    xlabel('Date')
    ylabel('GWL [m asl]')
    s.Position(3) = s.Position(3)*0.91;
    legend('observed','simulated median','Position',[0.8615,0.8608,0.12,0.0647]);
    datetick('x','yyyy-mm','keepticks')
    
    %Error Box
    elapsedtime = time2-time1;
    
    dim = [0.8615,0.25,0.10,0.58];
    str1 = ["HiddenSize","rH","T","Tsin","ID P","ID rH","ID T","ID Tsin","ID GWLt-1","FD"];
    str2 = string(bestKonfig{pp,2:11});
    str2(ismissing(str2))="-";str2(2:4) = strrep(str2(2:4),'0','No');str2(2:4) = strrep(str2(2:4),'1','Yes');
    str1_2 = {["NSE [ ]" ],["R² [ ]"],["RMSE [m]" ],["rRMSE [%]"],["Bias [m]" ],["rBias [%]"],["PI12 [ ]"],["PI12op [ ]"],[" "]};
    str2_2 = {sprintf('%0.2f\n%0.2f\n%0.2f\n%0.2f\n%0.2f\n%0.2f\n%0.2f\n%0.2f\n',...
        median(Forecast_Errors(1,:)),median(Forecast_Errors(2,:)),...
        median(Forecast_Errors(3,:)),median(Forecast_Errors(4,:)),...
        median(Forecast_Errors(5,:)),median(Forecast_Errors(6,:)),...
        median(Forecast_Errors(7,:)),median(Forecast_Errors(8,:)))};
    annotation('textbox',dim,'String',[str1_2,str1]);
    annotation('textbox',dim+[0.06,0,0,0],'String',[str2_2,str2],'EdgeColor','none');
    
    print('-dpng','-r300',strcat(".\Forecast_NARX_seq2val_12steps_",Well_ID))
    %%
    % close all
    save(strcat(".\Forecast_NARX_seq2val_12steps_Workspace_",Well_ID));
end


%%
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