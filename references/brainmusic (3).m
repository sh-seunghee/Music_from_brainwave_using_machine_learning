function [nmat]=brainmusic(data,sf)
% Code Summary for working in UESTC
% Contact: Jing Lu, e-mail: lujing@uestc.edu.cn
%clear all;
%fn=input('Data filename:','s');
%fn=strcat(fn,'.mat');
%sf=input('sample rate:')
%sf=500;%采样率
%load(fn);%载入脑电数据
% load('ratNREM.mat');
% data=data;
% data=eval(fn);
% data=data(3001:13000);
dl=length(data);%计算数据长度

%------------------判断数据值的符号-----------------
sig=sign(data);
pot=find(sig);
for i=1:dl
    if sig(i)>=0
        noteon(i)=1;
    else noteon(i)=0;%noteon为1时是正值，为0是负值
    end
end
%-----------------寻找过零基线的点-------------------
for i=2:dl
    if noteon(i-1)==0&noteon(i)==1
        t(i)=1;
    else t(i)=0;
    end
end
%---------------计算周期----------------------------
tfb=find(t);
tfb1=tfb(1:end-1);
tfb2=tfb(2:end);
ntime=tfb2-tfb1;%每个周期的时间长度
nl=length(ntime);%音符的个数

dur=ntime/sf;%dur-音的持续长度，单位s
dfre=sf./ntime;%计算频率

%---------------计算振幅----------------------------
for i=1:nl
    nal(i)=max(data(tfb(i):tfb(i+1)))-min(data(tfb(i):tfb(i+1)));%每个周期的振幅
%     nal(i)=nal(i)*0.5;
    nmark(i)=max(data(tfb(i):tfb(i+1)));%标记
end

%---------------计算音高 pit-------------------------------
nx=[];
for i=1:length(ntime)
    if nal(i)>1&&nal(i)<200
        if dfre(i)>10
            alpha=1.50;
            pit(i)=96-round(40/alpha*log10(nal(i)));
        else alpha=0.48;
            pit(i)=109-round((40/alpha*log10(nal(i)))/190*84);
        end
    elseif nal(i)<=1
        pit(i)=96;
    elseif nal(i)>=200
        pit(i)=24;
    end
        
    nx0=[];
    for j=1:ntime(i)
        nx0=[nx0 pit(i)];
    end
    nx=[nx nx0];
end
nx=[zeros(1,tfb(1)-1) nx zeros(1,dl-tfb(end)+1)];


%------------计算音量 vol-----------------------------------
for i=1:length(ntime)
     w=data(tfb1(i):tfb2(i));
     ap(i)=mean(w.^2);
end
apd=diff(ap);
apd=[1 apd];
apd1=log10(abs(apd)).*sign(apd);
vol=64+apd1*16;
for i=1:length(ntime)
    if vol(i)>=127
        vol(i)=127;
    elseif vol(i)<1
        vol(i)=1;
    end
end
vol=round(vol);%音强

% %--------------MIDI音乐制作-----------------------------
 nmat=zeros(nl,7);
 ts(1)=0;
 for i=1:nl
    ts(i+1)=ts(i)+dur(i);
 end
 tempo=120;
 tb=ts*tempo/60;
 db=dur*tempo/60;
 nmat(:,1)=tb(1:end-1);
 nmat(:,2)=db;
 nmat(:,3)=1;
 nmat(:,4)=pit;
 nmat(:,5)=vol;
 nmat(:,6)=ts(1:end-1);
 nmat(:,7)=dur;
%writemidi(nmat,['fn.mid'],120,120,4,4);

% figure;
% pianoroll(nmat,'num','vel','sec','g');

