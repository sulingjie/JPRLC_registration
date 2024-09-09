clc; clear; close all;
%----------------  load data  ---------------------%
N = 4; % number of point clouds
X = arrayfun(@(j) dlmread(sprintf('./blade%d.txt',j),' ')',[1:N]','uniformoutput',false);

%---------------- registration  -------------------%
% % initialization
M = 1000; % number of GMM components

% initialization of centroids of GMM
az = 2*pi*rand(1,M);
el = 2*pi*rand(1,M);
%points on a unit sphere
Y0 = [cos(az).*cos(el); sin(el) ; sin(az).*cos(el) ];
Y0 = 50 * Y0 + mean(X{1},2);

sigma0 = repmat(1000,M,1); % variances of G
R0 = repmat({eye(3)}, N, 1); % initial rotation matrix
t0 = cellfun(@(X) -mean(X,2)+mean(Y0,2),X,'UniformOutput',false); % initial translation vector
IterNum = 100; % iteration number
lambda = 0.1; % weight of local consistency term
[TX, R_result, t_result, Y] = JPRLC(X,Y0,sigma0,R0,t0,IterNum,lambda,0.1);
%---------------  show results  -------------------%


figure; hold on;
scatter3(Y(1,:),Y(2,:),Y(3,:),'k');
hold off;


figure; hold on;
cellfun(@(X) scatter3(X(1,:),X(2,:),X(3,:)), TX, 'UniformOutput', false);
hold off;
