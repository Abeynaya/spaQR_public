clc
clear all

rand('state',45)
% LS system for variable coefficient poisson
s = 128; % Change size 

An =sparse( diag(2*ones(s,1)) - diag(ones(s-1,1),1) - diag(ones(s-1,1),-1));
I = sparse(eye(s,s));
L = kron(An,I)+kron(I,An);

nu1 = s;
nz1 = s+1;

m = nu1^2;
nu = nu1^2;
nz = nz1^2;
n = nu+nz;

% Au = sparse(m,nu);
% Az = sparse(m,nz);

Au = L; % set all z_ij to 1


U = ones([nu1,nu1]);
step = 1; % Change step to change aspect ratio of matrix
% U(2:step:nu1,:)=0;

U = [zeros(1,nu1); U; zeros(1,nu1)];
U = [zeros(nu1+2,1) U zeros(nu1+2,1)];
I = zeros(ceil(m*4),1);
J = zeros(ceil(m*4),1);
S = zeros(ceil(m*4),1);

uij = U(2:nu1+1, 2:end-1);
ui1j = U(2:nu1+1, 3:end);
uij1 = U(3:nu1+2, 2:end-1);
ui_1j = U(2:nu1+1, 1:end-2);
uij_1 = U(1:nu1, 2:end-1);

b0 = -uij + 0.5*ui1j + 0.5*uij1;
b1 = -uij + 0.5*uij1 + 0.5*ui_1j;
b2 = -uij + 0.5*ui1j + 0.5*uij_1;
b3 = -uij + 0.5*ui_1j + 0.5*uij_1;
counter = 0;
for k=1:nu1
    B = [b3(k,:)' b2(k,:)' b1(k,:)' b0(k,:)'];
    d = [0 1 nz1 nz1+1];
    At = spdiags(B,d,nu1, 2*nz1);
    [i, j, sn] = find(At);
    i = i + (k-1)*nu1;
    j = j + (k-1)*nz1;
    I(counter+1: counter+size(i,1)) = i;
    J(counter+1: counter+size(i,1)) = j;
    S(counter+1: counter+size(i,1)) = sn;
    counter = counter + size(i,1);
end

Az = sparse(I(1:counter),J(1:counter),S(1:counter),m, nz);
Az(:, ~any(Az,1)) = [];
A = [Au Az];
B = A';
nnz(A)
size(B)
disp('aspect ratio')

ar = size(B,1)/size(B,2)
%%
file = strcat('../mats/invpoi/2d/invpoi_2d_1_',num2str(s),'.mm');

mmwrite(file,B); 




