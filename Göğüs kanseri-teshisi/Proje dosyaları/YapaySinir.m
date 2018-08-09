
function[net,ye,yv,MAPE,R2] = YapaySinir( input, target ,training_rate,n1,n2,lrate)                                                  
%UNT�TLED2 Summary of this function goes here
%   Detailed explanation goes here
%n1 ve n2 gizli katmandaki n�ron say�s�
%lrate =�grenme katsay�s�
 noofdata=size(input,1);
ntd=round(noofdata*training_rate);
xt=input(1:ntd,:); %egitim verileri    training_rate 

xv=input(ntd+1:end,:);

yt=target(1:ntd);

yv=target(ntd+1:end);

% verilerin xt,xv,ty,transpozunu almam�z gerekiyor
xt=xt';
xv=xv';
yt=yt'; %��kt� verileri
yv=yv';%��kt� verileri
%yukar�da verilerin transpozunu ald�k
%�imdi girdi verilerinin normalize edilmesi gerekiyor
%xtn=egitim i�in yani tarning i�in normall girdi
%xvn=sonuc i�in normal girdi

xtn=mapminmax(xt);
xvn=mapminmax(xv);

%ytn normalize edilmi�      training_rate

[ytn, ps]=mapminmax(yt); %ps normalizasyonun nas�l yap�ld�g� ile ilgili bilgiyi saklar
%ag� olu�turuyoruz
%newff komutu
%girdileri s�rayla �unlard�r (girdi,��kt�,n�r�n say�s�,transfer fonksiyonu,algoritma ne kullan�lacak)
net=newff(xtn,ytn,[n1,n2], {  }, 'trainlm');
net.trainParam.lr=lrate;
net.trainParam.epochs=10000;  %itersyon say�s�n� veriyoruz
net.trainParam.goal=1e-100000000;
net.trainParam.showWindow = false;

%ag� egitme net ag�n� egit netin �zerine ekle

net=train(net,xtn,ytn); %xtn =egitim girdisi ytn=egitim hedefi
%yen=validation girdileirin gir normalizasyon ��kt�lar�n� elde et 
%normalize haldeki ��kt�lar
yen=sim(net,xvn);%tesssssssssssssssssssss
%yvn=normalize vn   vn=normal girdi
%ye=normalize olmayan ��kt�lar
ye=mapminmax('reverse',yen,ps);
ye=ye';
yv=yv';
MAPE=mean((abs(ye-yv))./yv);
SStotal=sum((yv-mean(yv)).^2);
SSeror=sum((ye-yv).^2);
R2=1-SSeror/SStotal;

end

