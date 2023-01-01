clc; clear all;
% create look up table in driver program
lut_cos = zeros(25,127); % to be optimised later
lut_sin = zeros(25,127);
for j = 1:25
    for k = 1:127 % to be optimised later
        lut_cos(j,k) = int8(floor(cos((j+4)*k*pi/64)*(2^7-1))); % keep this 16 bits
        lut_sin(j,k) = int8(floor(sin((j+4)*k*pi/64)*(2^7-1)));
    end
end
lut_cos = cast(lut_cos, 'int32');
lut_sin = cast(lut_sin, 'int32');

% create matched filter and quantize it (in driver program)
rolloff = 0.35; % Filter rolloff
span = 6;       % Filter span
sps = 16;       % Samples per symbol 
b = rcosdesign(rolloff, span, sps);
b = int16(floor(b*(2^15-1)/max(b)));  
b = cast(b,'int32');

% %%%%%%%%%%%%%%%%%%%%%% Evaluation for Problem 1 %%%%%%%%%%%%%%%%%%%%%%%%%
% T = zeros(80,100);
% for sim = 1:100
%     S = generate();
%     p = transmitter(S);
%     for i = 0:79
%             q = myChannel(100, 0, 0, 0, p, i);
%             f = myReceiver(q,b);
%             [~,t,~,~] = myDFT(f,lut_cos,lut_sin);
%             T(i+1,sim) = t;
%     end
% end
% % Convert T into time
% for i = 0:79
%     for j = 1:100
%         T(i+1,j) = (T(i+1,j)-40)*0.0625;
%     end
% end
%%Finding Mean and Standard Deviation of Time Estimate
% for i = 1:80
%     Tt_mean(i) = sum(T,2)/100;
%     std_t(i) = std(T(i,:));
% end

% %%%%%%%%%%%%%%%%%%%%%%%% Evaluation for Problem 2 %%%%%%%%%%%%%%%%%%%%%%%
% F = zeros(25,100);
% for sim = 1:100
%     S = generate();
%     p = transmitter(S);
%     for i = -1500:125:1500
%             q = myChannel(100, 0, i, 0, p, 0);
%             f = myReceiver(q,b);
%             [~,~,freq,~] = myDFT(f,lut_cos,lut_sin);
%             F((i+1625)/125,sim) = freq;
%     end
% end
% % Convert F into frequency
% for i = 1:25
%     for j = 1:100
%         F(i,j) = (F(i,j) - 12)*125;
%     end
% end
% %Finding Mean and Standard Deviation
% for i = 1:25
%     Tf_mean(i) = sum(F,2)/100;
%     std_f(i) = std(F(i,:));
% end

%%%%%%%%%%%%%%%% Evaluation for Problem 3 part(a), (b), (c) %%%%%%%%%%%%%%%
% FER = zeros(37,1);
% sim = 1;
% for i = -3:0.5:15
%      while( sim<1001)
%             % Waiting For atmost 10,000 simulations to get 50 frame errors.
%             S = generate();
%             p = transmitter(S);
%             q = myChannel(i, 4, 62.5, 0, p, 40); 
%             %myChannel(snr, delay_offset, freq_offset, randphase, signal, delay_sample)
%             
%             % 40 samples correspond to 0 delay. This is for part (a).
%             
%             % Replace frequency offset by 600 and delay offset by 4 for
%             % part (b).
%             
%             % Replace frequency offset by 62.5 and delay offset by 8 for
%             % part (c). Can't figure out how to create a delay of 2.5/80
%             % msec.
%             [f, max_offset, I, S1] = myReceiver(q,b);
%             [~, time_index, freq, retr, ~] = myDFT(f,lut_cos,lut_sin);  
%             Tf(2*i+7,sim) = freq;
%             Tt(2*i+7,sim) = time_index;
%             max_offset_array(2*i+7,sim) = max_offset;
%             BER(2*i+7,sim) = myComparator(retr,S);
%             if BER(2*i+7, sim) > 0
%                 FER(2*i+7) = FER(2*i+7) +1;
%             end
%             sim = sim+1;
%      end
%      FER(2*i+7) = FER(2*i+7)/sim;
%      sim = 1;
% end
% %Convert Tf and Tt arrays into time and frequencies
% for i = 1:37
%     for j = 1:length(Tt(i,:))
%         Tt1(i,j) = (Tt(i,j)-40)*0.0625 + 0.0625/16*(4);
%     end
% end
% for i = 1:37
%     for j = 1:length(Tf(i,:))
%         Tf1(i,j) = (Tf(i,j)-12)*125;
%     end
% end 
% %Finding Mean and Standard Deviation
%  for i = 1:37
%     Tt_mean(i) = sum(Tt1(i,:))/length(Tt1(i,:)); % Mean Of Time estimate
%     Tf_mean(i) = sum(Tf1(i,:))/length(Tf1(i,:)); % Mean Of Frequency estimate
%      std_t(i) = std(Tt(i,:)); % Standard Deviation of Time estimate
%     std_f(i) = std(Tf1(i,:)); % Standard Deviation of Frequency estimate
%  end
% Plotting FER and BER Curves
% for i = 1:37
%     SNR_array(i) = (i-7)/2.0;
%     BER_mean(i) = sum(BER(i,:))/length(BER(i,:)); % Finding mean BER over all simulations
% end
% semilogy(SNR_array, FER);
% hold on
%  plot(SNR_array, BER_mean);
% hold off
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TEST CODE
% for i = 1:3
% S = generate();
% T = transmitter(S);
% R = myChannel(15, 4,125, 0, T, 40);
% [F, max_offset, I, S1] = myReceiver(R,b);
% [F_dft, time_index, freq_index, retr, dft_mag] = myDFT(F,lut_cos,lut_sin);
% [BER(i), decoded] = myComparator(retr, S);
% end
% plot(BER(i));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTION DEFINITIONS %%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate Frame Of size 800
function [S] = generate()
    framesize = 800;  % set framesize here
    S = zeros(framesize,1);
    for i = 1:128
        S(i) = 1;
    end
    for i = 129:136
        S(i) = 0;
    end
    for i = 137:framesize
        S(i) = randi([0,1]);
    end
end

% Design Of Transmitter
function [T,T1,T2,b] = transmitter(rawData)
       framesize= 800;                
       T1 = zeros(framesize,1);
       for j = 1:framesize
           % pi/4 BPSK
           T1(j) = (1-2*rawData(j))*(cos(pi*(j-1)/4.0)+sin(pi*(j-1)/4.0)*1i); % Modulated Data
       end
       % Upsample here
       T2 = upsample(T1,16);
       % RRC Filtering
       rolloff = 0.35; % Filter rolloff
       span = 6;       % Filter span
       sps = 16;       % Samples per symbol 
       b = rcosdesign(rolloff, span, sps);
       T = conv(T2(:,1),b); 
       span = 48;
       T = T(span+1:end-span);
end
 
 % Design Of Channel
function [R] = myChannel(snr, delay_offset, freq_offset, randphase, signal, delay_sample)
          framesize = 800;                         
          fS = 16000; % sampling frequency
          R = zeros((framesize+80)*16,1);
          delay = delay_offset+16*delay_sample;
          % delay_offset is a value between 1 and 16
          % delay_sample is a value between 0 and 79
          for j = 1:length(signal)
              R(j+delay) = signal(j)*exp(1i*(2*pi*freq_offset*(j-1)/(16*fS) + randphase));
          end
          R = awgn(R,snr,'measured');
end

% Fixed Point Receiver
function [F, max_offset, I, S1] = myReceiver(R,b)
    % Quantize R
    maxr = max(real(R));
    maxi = max(imag(R));
    for k = 1:880*16
        S1(k,1) = int16(floor(real(R(k))*(2^15-1)/maxr));
        S1(k,2) = int16(floor(imag(R(k))*(2^15-1)/maxi));
        % S1 is a 32 bit register with 16 bits each for real and imaginary
    end

  F = zeros(880,2);
  F = int32(F);           % 64 bit register
  sum = int64(0);         % 64 bit register
  sum_prev = int64(0);    % 64 bit regiser
  max_offset = int16(1);  % 16 bit register

% Convolve
for n = 1:886*16
    I(n,1) = 0;
    I(n,2) = 0;
    for k = 1 : 880*16
        if(n - k + 1 >= 1 && n - k + 1 <= length(b))
            I(n,1) = I(n,1) + (cast(S1(k,1),'int32'))*b(n - k + 1);
            I(n,2) = I(n,2) + (cast(S1(k,2),'int32'))*b(n - k + 1);
        end
    end
end 

  span = 48;
  for i = 1:height(I)-2*span
      I(i,1) = I(i+span,1);
      I(i,2) = I(i+span,2);
  end
  I(height(I)-2*span+1:height(I),:) = [];
  I = bitsra(I,16);

  % Finding Correct Sampling Time
  for offset = 1:16
      for j = 0:39
          k = 16*j+offset;
          sum = sum + I(k,1)*I(k,1)+I(k,2)*I(k,2); 
      end
      if sum > sum_prev
          sum_prev = sum;
          max_offset = offset;
      end
      sum = 0;
  end
  % Undersampled Signal
  for j = 1:880
      F(j,1) = I(16*(j-1) + max_offset,1);
      F(j,2) = I(16*(j-1) + max_offset,2);
  end
end

% DFT in Fixed Point
function [F_dft, time_index, freq_index, retr, dft_mag] = myDFT(F,lut_cos,lut_sin)    
    F_dft = zeros(25,2);
    dft_mag = zeros(25,1);
    max_mag = zeros(80,1);    
    for l = 1:80
        F1 = cast(F(l:l+127,:),'int32');
        for j = 1:25
        for k = 1:127
            F_dft(j,1) = F_dft(j,1) + lut_cos(j,k)*F1(k+1,1) + lut_sin(j,k)*F1(k+1,2); 
            F_dft(j,2) = F_dft(j,2) - lut_sin(j,k)*F1(k+1,1) + lut_cos(j,k)*F1(k+1,2);
        end
        
        % Add the term corresponding to 1 here
        F_dft(j,1) = F_dft(j,1) + F1(1,1);
        F_dft(j,2) = F_dft(j,2) + F1(1,2);
        
        % Normalise by order of DFT
         F_dft(j,1) = bitsra(F_dft(j,1),7);
         F_dft(j,2) = bitsra(F_dft(j,2),7);
        
        % Choose max from the 25 coefficients and store it back in F_dft
        dft_mag(j,1) = F_dft(j,1)*F_dft(j,1) + F_dft(j,2)*F_dft(j,2); % 64 bits
        end
        max_mag(l,1) = max(dft_mag); % 64 bits
    end  

    [~,time_index] = max(max_mag); % Calculates beginning point of signal
    F1 = cast(F(time_index:time_index+127,:),'int32');
    F_dft = zeros(25,2); % F_dft register clear

    for j = 1:25
        for k = 1:127
            F_dft(j,1) = F_dft(j,1) + lut_cos(j,k)*F1(k+1,1) + lut_sin(j,k)*F1(k+1,2); 
            F_dft(j,2) = F_dft(j,2) - lut_sin(j,k)*F1(k+1,1) + lut_cos(j,k)*F1(k+1,2);
        end

        % Add the term corresponding to 1 here
        F_dft(j,1) = F_dft(j,1) + F1(1,1);
        F_dft(j,2) = F_dft(j,2) + F1(1,2);
        
        % Choose max from the 25 coefficients and store it back in F_dft
        dft_mag(j,1) = F_dft(j,1)*F_dft(j,1) + F_dft(j,2)*F_dft(j,2); % 64 bits    
    end

    [~,freq_index] = max(dft_mag); % Calculates frequency offset estimate
    
    % Retrieved Signal
    retr = cast(F(time_index:time_index+799,:),'int32');
    for j = 1:800
        if mod(j-1,128) ~= 0
           retr(j,1)= retr(j,1)*lut_cos(freq_index,mod(j-1,128)) + retr(j,2)*lut_sin(freq_index,mod(j-1,128));
           retr(j,2)= retr(j,2)*lut_cos(freq_index,mod(j-1,128)) - retr(j,1)*lut_sin(freq_index,mod(j-1,128));
        else 
           retr(j,1)= retr(j,1);
           retr(j,2)= retr(j,2);
        end
    end
end

% BER and FER Computation
function [BER, decoded] = myComparator(retr, S)
    BER = 0;
    decoded = zeros(800,1);
    % Decode the return signal
    for i = 1:800
        if (retr(i,1) > 0)
            decoded(i) = 0;
        else 
            decoded (i) = 1;
        end
    end
    for i = 1:800
        if decoded(i) ~= S(i)
            BER= BER+1;
        end
    end
    BER = BER/800;
end