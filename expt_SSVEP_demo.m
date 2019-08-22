clear
% rng('default')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%% Load signal package to call dct function. Comment this line if using MATLAB.
pkg load signal

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load data corresponding to Subject 27. For more details of dataset, please refer:
% Y. Wang, X. Chen, X. Gao and S. Gao, "A Benchmark Dataset for SSVEP-Based 
% Brain–Computer Interfaces," in IEEE Transactions on Neural Systems and 
% Rehabilitation Engineering, vol. 25, no. 10, pp. 1746-1752, Oct. 2017.
s =27;
load S27;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Data corresponding to every subject (total 35 subjects) has 
% 64 channels x 1500 points x 40 trials x 6 blocks.
% For analysis, we have used nine electrodes over the parietal and occipital areas 
% (Pz, PO5, POz, PO4, PO6, O1, Oz, O2 and CB2) i.e. 
% Electrodes number: [48 54 55 56 57 58 61 62 63]
e = [48 54 55 56 57 58 61 62 63];
data = data(e,:,:,:);
shape = size(data);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Frequency information corresponding to 40 frequencies:
load Freq_Phase

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initial data required to perform Canonical Correlation Analysis (CCA) 
Fs = 250 ;              % Sampling Rate
H = 2;                  % Number of Harmonics for reference construction
t_length = 6;           % Analysis window length (in seconds)
TW = 0.5:0.5:t_length;
len = length(TW);   
TW_p = round(TW*Fs);
n_cross = 6;              % To perform leave-one-out cross validation
n_st_f = length(freqs);   % Number of stimulus frequencies
labels = [1:1:n_st_f];

sc = cell(1,40);          % Reference Signal Construction with sine-cosine waveforms
for i = 1 : length(freqs)
    sc{i} = myrefsig(freqs(i), H, t_length, Fs);
end


%% Output Accuracy initialization and other initializations
Runs = 1;              % Total number of runs with different realization of Phi
% accuracy_CCA = cell(Runs,1);
N = 250;
CR = 70;
M = N - N*CR/100 ;
number_of_segments = floor(shape(2)/N);
% The block partition for STSVB 
block_length = 25;
block_start_loc = 1 : block_length : N;
block_ind = reshape(ones(block_length,1)*(1:N),N*block_length,1);

for run = 1 : Runs
  
    run
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%% COMPRESSIVE SENSING OF EEG DATA %%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% Sparse binary full rank sensing matrix generation with two entries of 1's 
    % at random locations in each column while rest of the entries are 0.
    while(1)
        [Phi,flag] = genPhi(M,N, 2);
        if flag == 1, break; end;
    end
    
    % Use DCT dictionary matrix
    A=zeros(M,N);
    for k=1:M
        A(k,:)=dct(Phi(k,:));
    end
    
    X_hat_LSTSVB = zeros(shape);
    it=10;
    
    for j = 1 : number_of_segments   % Segment-wise Processing
        
        for i = 1  : shape(4) % Trial-wise processing
            
            for k = 1 : shape(3) % Frequency-wise processing
              
                %%%%%%%%%%%%%%% Compressively Sensed Data %%%%%%%%%%%%%%%
                y = Phi * data(:, (j-1)*N+1 : j*N, k,i)';
                
                %%%%%%%%%%%%%%%%% Recovery using STSVB %%%%%%%%%%%%%%%%%
                Result_LSTSVB = stsvb_demo(A,y,block_start_loc,1,0,it);
                signal_hat_LSTSVB = idct(Result_LSTSVB.x);
                X_hat_LSTSVB(:,(j-1)*N+1:j*N, k,i) = (signal_hat_LSTSVB)';
                fprintf('STSVB-L Segment %d out of %d\n',j,number_of_segments)
                
            end
        end
        
    end
    
    n_correct = zeros(len,2);        % Correct frequency detection initialization
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%% CANONICAL CORRELATION ANALYSIS %%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%% Leave-one-run-out cross-validation %%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for cros = 1 : n_cross     
      
        idx_testdata = cros;
        idx_traindata = 1 : n_cross;
        idx_traindata(idx_testdata) = [];
        
        for tw_length = 1 : len
             fprintf('Original EEG, Subject %d, CCA Processing TW %fs, No.crossvalidation %d \n', s, TW(tw_length),cros);
             idx = zeros(1,n_st_f);
             % Recognize SSVEP using CCA
             for j = 1 : n_st_f
                 r_xy = [];
                 for i = 1 : n_st_f
                     [~,~,r] = mycca(data(:,1:TW_p(tw_length),j,idx_testdata),sc{i}(:,1:TW_p(tw_length)));
                     r_xy = [r_xy r(1)];
                 end
                 [~,idx(j)] = max(r_xy);
             end
             is_correct = (idx == labels);
             n_correct(tw_length,1) = n_correct(tw_length,1) + length(find(is_correct==1));
            
            
            fprintf('STSVB-L EEG, Subject %d, CCA Processing TW %fs, No.crossvalidation %d \n', s, TW(tw_length),cros);
            idx = zeros(1,n_st_f);
            % Recognize SSVEP using CCA
            for j = 1 : n_st_f
                r_xy = [];
                for i = 1 : n_st_f
                    [~,~,r] = mycca(X_hat_LSTSVB(:,1:TW_p(tw_length),j,idx_testdata),sc{i}(:,1:TW_p(tw_length)));
                    r_xy = [r_xy r(1)];
                end
                [~,idx(j)] = max(r_xy);
            end
            is_correct = (idx == labels);
            n_correct(tw_length,2) = n_correct(tw_length,2) + length(find(is_correct==1));
            
         end  
         
         accuracy_CCA = 100 * n_correct / (n_st_f * n_cross);
     end
     
     figure
     method = {'original','STSVB-L'}
     color={'b-*','r-o'};
     for mth=1:2
        plot(TW,accuracy_CCA(:,mth),color{mth},'LineWidth',1);
        hold on
     end
     xlabel('Time window (s)');
     ylabel('Accuracy (%)');
     xlim([0.25 6.25]);
     set(gca,'xtick',0.5:0.5:6,'xticklabel',0.5:0.5:6);
     ylim([0 100]);
     title('CCA for SSVEP Recognition');
     grid;
     h=legend(method);
     set(h,'Location','SouthEast');
     
end