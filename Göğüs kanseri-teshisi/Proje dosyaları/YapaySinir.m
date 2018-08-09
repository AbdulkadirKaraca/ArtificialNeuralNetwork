
function[net,ye,yv,MAPE,R2] = YapaySinir( input, target ,training_rate,n1,n2,lrate)                                                  
%UNTÝTLED2 Summary of this function goes here
%   Detailed explanation goes here
%n1 ve n2 gizli katmandaki nöron sayýsý
%lrate =ögrenme katsayýsý
 noofdata=size(input,1);
ntd=round(noofdata*training_rate);
xt=input(1:ntd,:); %egitim verileri    training_rate 

xv=input(ntd+1:end,:);

yt=target(1:ntd);

yv=target(ntd+1:end);

% verilerin xt,xv,ty,transpozunu almamýz gerekiyor
xt=xt';
xv=xv';
yt=yt'; %çýktý verileri
yv=yv';%çýktý verileri
%yukarýda verilerin transpozunu aldýk
%þimdi girdi verilerinin normalize edilmesi gerekiyor
%xtn=egitim için yani tarning için normall girdi
%xvn=sonuc için normal girdi

xtn=mapminmax(xt);
xvn=mapminmax(xv);

%ytn normalize edilmiþ      training_rate

[ytn, ps]=mapminmax(yt); %ps normalizasyonun nasýl yapýldýgý ile ilgili bilgiyi saklar
%agý oluþturuyoruz
%newff komutu
%girdileri sýrayla þunlardýr (girdi,çýktý,nörön sayýsý,transfer fonksiyonu,algoritma ne kullanýlacak)
net=newff(xtn,ytn,[n1,n2], {  }, 'trainlm');
net.trainParam.lr=lrate;
net.trainParam.epochs=10000;  %itersyon sayýsýný veriyoruz
net.trainParam.goal=1e-100000000;
net.trainParam.showWindow = false;

%agý egitme net agýný egit netin üzerine ekle

net=train(net,xtn,ytn); %xtn =egitim girdisi ytn=egitim hedefi
%yen=validation girdileirin gir normalizasyon çýktýlarýný elde et 
%normalize haldeki çýktýlar
yen=sim(net,xvn);%tesssssssssssssssssssss
%yvn=normalize vn   vn=normal girdi
%ye=normalize olmayan çýktýlar
ye=mapminmax('reverse',yen,ps);
ye=ye';
yv=yv';
MAPE=mean((abs(ye-yv))./yv);
SStotal=sum((yv-mean(yv)).^2);
SSeror=sum((ye-yv).^2);
R2=1-SSeror/SStotal;

end

