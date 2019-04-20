%%%%%% Implementation of FOS for Non-Linear Systems %%%%%%
%%%%%% Using System Input - Uniformly Distributed Noise %%%%%%
%%%%%% N1 : No. of Input data points %%%%%%
clc;
clear all;
N1 = 3000;
%%% Uniformly Distributed Pseudorandom Numbers %%%
rng(123)
x = rand(1,3000);

%%%%%% Non-Linear System Identification %%%%%%
%%%%%% R: Memory of the system
%%%%%% L: Memory of input, K: Memory of output
%%%%%% System-1 Lag Matrix
K1 = [6 8 7 10 14 12];
L1 = [5 3 10 12 6 8];
%%%%%% System-2 Lag Matrix
% K1 = [3 5 7 4 3 4];
% L1 = [12 15 13 8 13 11];
%%%%%% System-3 Lag Matrix
% K1 = [3 4 5 2 3 4];
% L1 = [5 7 10 3 5 7];
%%%%%% Order %%%%%%
Order = [2 2 2 2 3 3];
so = size(Order);
for cas = 1:so(2)
    %cas = 1;
    K = K1(cas);
    L = L1(cas);
    N0 = max(max([K1;L1]));
    R = K + L + 1;
    y(1:N1) = 0;
    for i = N0+1 : N1
        %%%%% Complex Difference Equations %%%%%
        % 1.
        y(i) = 1 + 0.78 * x(i) + 0.8 * x(i-1) + 0.1 * x(i-1) * y(i-1) - 0.4 * x(i) * x(i-1)+ 0.2 * x(i-4)*x(i-5);
        % 2.
        %         y(i) = (-0.8 + 0.5 * x(i-12) - 0.3 * x(i-5) + 0.2 * x(i-7) + 0.3 * y(i-3))*(1 + 0.4 * x(i-5) - 0.3 * x(i)^2);
        % 3.
        %         y(i) = ( 0.67 - 0.4 * x(i-5) + 0.6 * x(i-3) - 0.8 * y(i-2))*(1 - 0.2* x(i) + 0.7 * y(i-3))*(0.2 - 0.4 * x(i) + 0.3 * y(i-1));
        
    end
    w=y;
    e = rand(1,3000);
    percent = 0.0;
    y = y + sqrt(percent*var(y)/var(e))*(e-0.5);
    
    %%%%%%% PROCEDURE %%%%%%
    %%%%% Training Phase : N Data Points %%%%%
    N = 1000;
    %%%%%%% O: Order of volterra series for approximation of nonlinear system
    O = Order(cas);
    %%%%%% M1: No. of Oth order Candidates; M: Total no. of candidates
    M = 1;
    M1(1) = 1;
    for o = 1 : O
        M1(o+1) = (M1(o)*(R + o - 1))/o ;
        M = M + M1(o+1);
    end
    Pp(1:M,1:O) = -1;
    o = 0;
    for m = 2 : M
        if( m > sum(M1(1:o+1)) && o < O)
            o = o + 1;
            j(1:O) = -1;
            j(1:o) = 0;
        end
        Pp(m,:) = j;
        c = 0;
        for i1 = 1:o
            if(j(o) >= R-1 && j(i1) >= R-1 && i1 >1)
                if (c == 0 )
                    j(i1-1) = j(i1-1) + 1;
                end
                j(i1) = j(i1-1);
                c = 1;
            elseif(i1 == o && c == 0)
                j(o) = j(o) + 1;
            end
        end
    end
    alpha = 0;
    D(1) = 1;
    g(1) = sum(y(N0+1:N))/(N-N0);
    m1 = 1;
    pos = 1;
    ac = 0;
    Pal = 0;
    yn2 = sum(y(N0+1:N).^2)/(N-N0);
    diff = yn2 - g(1)^2 * D(1);
    %%%P(1,m): The g(m) terms
    %%%P(2,m): The D(m,m) terms
    while(m1 < M)
        P(1:O+2,m1) = -1;
        P(1,m1) = g(pos);
        P(2,m1) = D(pos,m1);
        P(3:2+O,m1) = Pp(pos,1:O);
        Pal(m1,1:m1-1) = alpha(pos,1:m1-1);
        m1 = m1 + 1;
        D1(1:M,m1) = 0;
        alpha = 0;
        for m = 2 : M
            j = Pp(m,:);
            if(sum(ismember(ac,m)) == 0)
                for r = 1 : m1
                    i1 = 1;
                    if ( r < m1 )
                        i = P(3:2+O,r);
                    else
                        i = j;
                    end
                    C = 0;
                    if(D1(m,r) == 0 || r == m1-1)
                        p = 1;
                        for i1 = 1:O
                            if(j(i1) <= L && j(i1) ~= -1)
                                p = p .* x(N0+1-j(i1):N-j(i1));
                            elseif( j(i1) ~= -1)
                                p = p .* y(N0+1-j(i1)+L:N-j(i1)+L);
                            end
                        end
                        if (r == m1)
                            C = p .* y(N0+1 : N);
                        end
                        i1 = 1;
                        while (i1 <= numel(i) && i(i1) ~= -1)
                            if(i(i1) <= L)
                                p = p .* x(N0+1-i(i1):N-i(i1));
                            else
                                p = p .* y(N0+1-i(i1)+L:N-i(i1)+L);
                            end
                            i1 = i1 + 1;
                        end
                        C = sum(C)/(N-N0);
                        D1(m,r) = sum(p)/(N-N0);
                    end
                    D(m,r) = D1(m,r);
                    if (r < m1)
                        if(r>1)
                            D(m,r) = D(m,r) - D(m,1:r-1)*Pal(r,1:r-1)';
                        end
                        alpha(m,r) = D(m,r)/P(2,r);
                    elseif(r == m1)
                        D(m,r) = D(m,r) - alpha(m,1:r-1).^2 * (P(2,1:r-1)');
                        C = C - alpha(m,1:r-1)*(P(2,1:r-1).*P(1,1:r-1))';
                        g(m) = C/D(m,r);
                    end
                end
            end
        end
        Q = g.^2 .* D(:,m1)';
        [Qmax,pos] = max(Q);
        a1 = size(ac);
        ac(a1(2)+1) = pos;
        %%%%%%%%%%%Statistical Check for continuing Search%%%%%%%%%%%
        %%%% SE: reduction in error provided by the (m+1)th term %%%%
        SE = g(pos)^2 * D(pos,m1);
        if(SE < (4/(N-N0))*diff || Qmax <=0 || diff < 0)
            break;
        end
        diff = diff - (g(pos)^2) * D(pos,m1);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    %%%%%%%%%%%%%%%%Calculating a(m)%%%%%%%%%%%%%%%%%%%%%%%%%%
    s = size(P);
    a(1:s(2)) = 0;
    for m = 1:s(2)
        v(m) = 1;
        for i = m : s(2)
            if (i > m )
                v(i) = 0;
                v(i) = v(i) - Pal(i,m:i-1) * v(m:i-1)';
            end
            a(m) = a(m) + P(1,i)*v(i);
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%% Obtaining Model Output %%%%%%%%%%%%%%%%%%%%%
    %%%% ymodel: output of the system using the chosen model %%%%
    for i = N0+1:N
        ymodel(i) = 0;
        for j = 1:s(2)
            Pmodel = 1;
            for k = 1:O
                d = P(k+2,j);
                if (d <= L && d~= -1)
                    Pmodel = Pmodel * x( i - d);
                elseif ( d ~= -1)
                    Pmodel = Pmodel * ymodel( i - d + L);
                else
                    break;
                end
            end
            ymodel(i) = ymodel(i) + a(j) * Pmodel;
        end
    end
    figure;
    pause(0.00001);
    frame_h = get(handle(gcf),'JavaFrame');
    set(frame_h,'Maximized',1);
    h=plot(1:N,y(1:N),'--',1:N,ymodel(1:N)); grid on;
    set(h(1),'Color',[1 0 0],'linewidth',5);
    set(h(2),'Color',[0 0 1],'linewidth',2);
    set(gca,'Color',[1 1 1]);
    hleg1 = legend('Actual Output of the System','Model Output');
    set(hleg1,'Location','NorthEast','FontSize',14);
    saveas(gcf,strcat('Train-M',cas+48,'-N100','.jpeg'));
    saveas(gcf,strcat('Train-M',cas+48,'-N100','.fig'));
    xlim([425 475]);
    saveas(gcf,strcat('Train-M',cas+48,'-N100-magnify','.jpeg'));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    mse(cas) = ((yn2-(P(1,1:s(2)).^2)*P(2,1:s(2))')/var(y(N0+1 : N)))*100;
    ideal_mse(cas) = ((sum((y(N0+1:N)-w(N0+1:N)).^2))/(N-N0))/(var(y(N0+1 :N)))*100;
    %%%% A: collection of am for different models generated %%%%
    A(cas,1:s(2)) = a(1:s(2));
    Pf(cas,1:O,1:s(2)) = P(3:2+O,1:s(2));
    clear Pal P D1 D M1 Q a alpha g ;
end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Selection Phase : Point#1001 to Point#2000
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for cas = 1:so(2)
    clear a;
    a = A(cas,:);
    clear P;
    if(cas<=4)
        O = 2;
    elseif(cas<=6)
        O = 3;
    else
        O = 4;
    end
    P = Pf(cas,1:O,:);
    s1 = size(Pf);
    clear ymodel;
    for i = 1000+N0:1000+N
        ymodel(i) = 0;
        for j = 1:s1(3)
            Pmodel = 1;
            for k = 1:O
                d = P(1,k,j);
                if (d <= L1(cas) && d~= -1)
                    Pmodel = Pmodel * x( i - d);
                elseif ( d ~= -1)
                    Pmodel = Pmodel * ymodel( i - d + L1(cas));
                end
            end
            ymodel(i) = ymodel(i) + a(j) * Pmodel;
        end
    end
    figure;
    pause(0.00001);
    frame_h = get(handle(gcf),'JavaFrame');
    set(frame_h,'Maximized',1);
    h=plot(1001+N0:1000+N,y(1001+N0:1000+N),'--',1001+N0:1000+N,ymodel(1001+N0:1000+N)); grid on;
    set(h(1),'Color',[1 0 0],'linewidth',5);
    set(h(2),'Color',[0 0 1],'linewidth',2);
    set(gca,'Color',[1 1 1]);
    hleg1 = legend('Actual Output of the System','Model Output');
    set(hleg1,'Location','NorthEast','FontSize',14);
    saveas(gcf,strcat('Test-M',cas+48,'-N100','.jpeg'));
    saveas(gcf,strcat('Test-M',cas+48,'-N100','.fig'));
    xlim([1425 1475]);
    saveas(gcf,strcat('Test-M',cas+48,'-N100-magni','.jpeg'));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    mse2(cas) = ((sum((ymodel(1001+N0:1000+N)-w(1001+N0:1000+N)).^2))/(N-N0))/(var(y(1001:1000+N0)))*100;
end
[mse_min posb]= min(mse2);
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Evaluation Phase (For minimum mse model) : Point#2001 to Point#3000
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear a P ymodel;
a = A(posb,:);
P = Pf(posb,:,:);
if(posb<=4)
    O = 2;
elseif(posb<=6)
    O = 3;
else
    O = 4;
end
s1 = size(P);
for i = 2001+N0:2000+N
    ymodel(i) = 0;
    for j = 1:s1(3)
        Pmodel = 1;
        for k = 1:O
            d = P(1,k,j);
            if (d <= L1(posb) && d~= -1)
                Pmodel = Pmodel * x( i - d);
            elseif ( d ~= -1)
                Pmodel = Pmodel * ymodel( i - d + L1(posb));
            else
                break;
            end
        end
        ymodel(i) = ymodel(i) + a(j) * Pmodel;
    end
end
figure;
pause(0.00001);
frame_h = get(handle(gcf),'JavaFrame');
set(frame_h,'Maximized',1);
h=plot(2001+N0:2000+N,y(2001+N0:2000+N),'--',2001+N0:2000+N,ymodel(2001+N0:2000+N)); grid on;
set(h(1),'Color',[1 0 0],'linewidth',5);
set(h(2),'Color',[0 0 1],'linewidth',2);
set(gca,'Color',[1 1 1]);
hleg1 = legend('Actual Output of the System','Model Output');
set(hleg1,'Location','NorthEast','FontSize',14);
saveas(gcf,strcat('Performance','-N100','.jpeg'));
saveas(gcf,strcat('Performance','-N100','.fig'));
xlim([2425 2475]);
saveas(gcf,strcat('Performance','-N100-magni','.jpeg'));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mse3 = ((sum((ymodel(2001+N0:2000+N)-w(2001+N0:2000+N)).^2))/(N-N0))/(var(y(2001+N0:2000+N)))*100;