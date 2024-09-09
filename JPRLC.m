function [ TX, R_result, t_result, Y ] = JPRLC( X, Y0, sigma0, R0, t0, IterNum, lambda, w)
% X: 只考虑点云内部的LC约束，不考虑点云之间的LC约束

sqe = @(Y,X) sum(bsxfun(@minus,permute(Y,[3 2 1]),permute(X,[2 3 1])).^2,3);

N = size(X,1);
M = size(Y0,2);


Y = Y0;
R = R0;
t = t0;
sigma2 = sigma0;
alpha = repmat((1-w)/M, M, 1); % 权重
Volumn = (max(X{1}(1,:))-min(X{1}(1,:)))*(max(X{1}(2,:))-min(X{1}(2,:)))*(max(X{1}(3,:))-min(X{1}(3,:)));

TX = cellfun(@(X,R,t) bsxfun(@plus,R*X,t),X,R,t,'uniformoutput',false);

allNj = [];
for j = 1:N
    allNj = [allNj ; size(X{j},2)];
end
%-------- 计算临近点 ---------%
nearIdx = cell(N,1); % 针对每个点保存该点临近点对应的点云索引和点索引，行：点云索引； 列：点索引
for j = 1:N
    Nj = allNj(j);
    nearIdx_j = cell(Nj,1);
    for i = 1:Nj
        nearIdx_j{i} = [];
    end
    nearIdx{j} = nearIdx_j;
end

for j = 1:N
    IDX = knnsearch(X{j}',X{j}', 'K', 10, 'Distance', 'euclidean');
    for i = 1:size(IDX,1)
        i_IDX = IDX(i,2:end);
        xi = X{j}(:,i);
        for k = 1:size(i_IDX,2)
            xk = X{j}(:,i_IDX(k));
            d = sqrt(sum((xi - xk).^2));
            if d < 5
                nearIdx{j}{i} = [nearIdx{j}{i} i_IDX(k)];
            end
        end
    end
end


R_result = cell(IterNum,1);
t_result = cell(IterNum,1);


for iter = 1:IterNum
    disp(iter);   
   %-----------------  Estep: posteriors -----------------%

   
   p = cellfun(@(X,R,t) sqe(R*X+t, Y),X,R,t,'uniformoutput',false);
   p = cellfun(@(p) bsxfun(@times, alpha.*power(2*pi*sigma2,-3/2), exp(bsxfun(@rdivide, -p, 2*sigma2))),p,'UniformOutput',false);
   p = cellfun(@(p) bsxfun(@rdivide, p, sum(p,1)+w/Volumn),p,'UniformOutput',false);
   if any(isnan(p{1}))
       disp('ERROR...');
   end
   %------------------  Mstep  -------------------%
   % % tj
   ux = zeros(3,N);
   uy = zeros(3,N);
   Np = zeros(N,1);
   for j = 1:N
       uxj = zeros(3,1);
       uxj_1 = zeros(3,1);
       uxj_2 = zeros(3,1);
       uyj = zeros(3,1);
       Npj = 0;
       Xj = X{j};
       Nj = size(Xj,2);
       Rj = R{j};
       pj = p{j};
       nearIdx_j = nearIdx{j};
       for i = 1:Nj
           xji = Xj(:,i);
           for m = 1:M
               p_jim = pj(m,i);
               ym = Y(:,m);
               sigma2_m = sigma2(m);
               uxj_1 = uxj_1 + p_jim/sigma2_m * xji;
               uyj = uyj + p_jim/sigma2_m * ym;
               Npj = Npj + p_jim/sigma2_m;
           end
       end

       % 计算 uxj_3
       for i = 1:Nj
           indexNP = nearIdx_j{i};
           xji = Xj(:,i);
           for index = 1:size(indexNP,2)
               b = indexNP(index);
               xjb = Xj(:,b);
               for m = 1:M
                   p_jim = pj(m,i);
                   p_jbm = pj(m,b);
                   sigma2_m = sigma2(m);
                   uxj_2 = uxj_2 + (p_jim-p_jbm)/sigma2_m*(xjb-xji);
               end
           end
       end
       
       uxj = uxj_1 + 0.5*lambda*uxj_2;
       
       ux(:,j) = uxj;
       uy(:,j) = uyj;
       Np(j) = Npj;
       
       t{j} = uyj./Npj - R{j}*(uxj./Npj); 
   end
   
   % Rj
   Xu = X;
   Yu = cell(N,1);
   for j = 1:N
       Xj = X{j};
       Xu{j} = bsxfun(@minus, Xj, ux(:,j)./Np(j));
       Yu{j} = bsxfun(@minus, Y, uy(:,j)./Np(j));
   end
   
   for j = 1:N
       Nj = allNj(j);
       Xu_j = Xu{j};
       Yu_j = Yu{j};
       H1 = zeros(3,3);
       pj = p{j};
       nearIdx_j = nearIdx{j};
       for i = 1:Nj
           xu_ji = Xu_j(:,i);
           for m = 1:M
               sigma2_m = sigma2(m);
               yu_mj = Yu_j(:,m);
               H1 = H1 + (pj(m,i)/sigma2_m)*xu_ji * yu_mj';
           end
       end
       
       
       H3 = zeros(3,3);
       for i = 1:Nj
           indexNP = nearIdx_j{i};
           xu_ji = Xu_j(:,i);
           for index = 1:size(indexNP,2)
               b = indexNP(index);
               xu_jb = Xu_j(:,b);
               for m = 1:M
                   p_jim = pj(m,i);
                   p_jbm = pj(m,b);
                   sigma2_m = sigma2(m);
                   yu_mj = Yu_j(:,m);
                   H3 = H3 + (p_jbm-p_jim)/sigma2_m*(xu_ji-xu_jb)*yu_mj';
               end
           end
       end
       
       H = H1 + 0.5*lambda*H3;
       if any(isnan(H))
           disp('ERROR...');
       elseif any(isinf(H))
           disp('ERROR...');
       else
           [U,~,V] = svd(H);
           R{j} = V*diag([1 1 det(V*U')])*U';
       end
   end
   
   TX = cellfun(@(X,R,t) bsxfun(@plus,R*X,t),X,R,t,'uniformoutput',false);
   

   for m = 1:M
       ym_1 = zeros(3,1);
       ym_2 = zeros(3,1);
       Npm = 0;
       sigma2_m = sigma2(m);
       for j = 1:N
           Nj = allNj(j);
           Rj = R{j};
           pj = p{j};
           tj = t{j};
           for i = 1:Nj
               xji = X{j}(:,i);
%                ym_1 = ym_1 + pj(m,i)/sigma2_m * Rj' * (xji-tj);
               ym_1 = ym_1 + pj(m,i)/sigma2_m * (Rj*xji+tj);
               
               Npm = Npm + pj(m,i)/sigma2_m;
           end
       end
       
       for j = 1:N
           Nj = allNj(j);
           nearIdx_j = nearIdx{j};
           for i = 1:Nj
               indexNP = nearIdx_j{i};
               for index = 1:size(indexNP,2)
                   b = indexNP(index);
                   ym_2 = ym_2 + (p{j}(m,i)-p{j}(m,b))/sigma2_m*...
                       R{j}*(X{j}(:,i)-X{j}(:,b));                  
               end
           end
       end
       
       Y(:,m) = (ym_1 - 0.5 * lambda * ym_2 )/ Npm;
   end

   % sigma2
   for m = 1:M
       sigma2_m1 = 0;
       sigma2_m2 = 0;
       Nm = 0; % 分母 
       ym = Y(:,m);
       for j = 1:N
           Nj = allNj(j);
           Rj = R{j};
           tj = t{j};
           for i = 1:Nj
               xji = X{j}(:,i);
               sigma2_m1 = sigma2_m1 + p{j}(m,i)*sum((Rj*xji+tj-ym).^2);
               Nm = Nm + p{j}(m,i);
           end
       end
       
       for j = 1:N
           Nj = allNj(j);
           nearIdx_j = nearIdx{j};
           for i = 1:Nj
               indexNP = nearIdx_j{i};
               xji = X{j}(:,i);
               for index = 1:size(indexNP,2)
                   b = indexNP(index);
                   p_jim = p{j}(m,i);
                   p_jbm = p{j}(m,b);
                   xjb = X{j}(:,b);
                   sigma2_m2 = sigma2_m2 + (p_jim-p_jbm)*...
                       (sum((Rj*xjb+tj-ym).^2)-sum((Rj*xji+tj-ym).^2)); 
               end
           end
       end
       sigma2_m = (sigma2_m1 + 0.5*lambda*sigma2_m2 )/(3*Nm);
       if sigma2_m > 0
           sigma2(m) = sigma2_m + 0.1; % avoid singularity
       end
   end
   % weight of local consistency term
   Nm = 0;
   for j = 1:N
       Nj = allNj(j);
       for i = 1:Nj
           for m = 1:M
               Nm = Nm + p{j}(m,i);
           end
       end
   end
   for m = 1:M
       alpha_m = 0;
       for j = 1:N
           Nj = allNj(j);
           for i = 1:Nj
               alpha_m = alpha_m + p{j}(m,i);
           end
       end
       alpha_m = (1-w)*alpha_m / (Nm + 0.1);
       alpha(m) = alpha_m;
   end
   
   % save current results
   R_result{iter} = R;
   t_result{iter} = t;
      
end
end

