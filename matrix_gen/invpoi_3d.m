% clc
% clear all

rand('state',45)
% LS system for 3D variable coefficient poisson
s = 32; % Choose size
d = 3; 

An =sparse( diag(2*ones(s,1)) - diag(ones(s-1,1),1) - diag(ones(s-1,1),-1));
I = sparse(eye(s,s));
L = kron(kron(An, I), I) + kron(kron(I, An), I) + kron(kron(I, I), An);

Au = L; % set all z_ij to 1

nu1 = s;
nz1 = s+1;

m = nu1^d;
nu = nu1^d;
nz = nz1^d;
n = nu+nz;

Az = sparse(m,nz);

U = ones([nu1+2,nu1+2, nu1+2]); % 3D grid
step = 1 ; % Change step to generate matrices with different aspect ratio
U(1:step:end-1,1:end-1,1:step:end-1)=0;


I = zeros(ceil(m*8),1);
J = zeros(ceil(m*8),1);
S = zeros(ceil(m*8),1);

uijk = U(2:end-1, 2:end-1, 2:end-1);
ui1jk = U(2:end-1, 3:end, 2:end-1); % i+1,j,j
uij1k = U(3:end, 2:end-1, 2:end-1); % i, j+1,k
uijk1 = U(2:end-1, 2:end-1, 3:end); % i,j, k+1

ui_1jk = U(2:end-1, 1:end-2, 2:end-1); %i-1,j,k
uij_1k = U(1:end-2, 2:end-1, 2:end-1); %i,j-1,k
uijk_1 = U(2:end-1, 2:end-1, 1:end-2); %i,j, k-1

b0 = -0.75*uijk + 0.25*ui1jk + 0.25*uij1k + 0.25*uijk1;
b1 = -0.75*uijk + 0.25*uij1k + 0.25*uijk1 + 0.25*ui_1jk;
b2 = -0.75*uijk + 0.25*ui1jk + 0.25*uijk1 + 0.25*uij_1k;
b3 = -0.75*uijk + 0.25*ui1jk + 0.25*uij1k + 0.25*uijk_1;

b4 = -0.75*uijk + 0.25*uijk1 + 0.25*ui_1jk + 0.25*uij_1k;
b5 = -0.75*uijk + 0.25*ui1jk + 0.25*uij_1k + 0.25*uijk_1;
b6 = -0.75*uijk + 0.25*uij1k + 0.25*ui_1jk + 0.25*uijk_1;
b7 = -0.75*uijk + 0.25*ui_1jk + 0.25*uij_1k + 0.25*uijk_1;

counter =0;
for p=1:nu1
    for k=1:nu1
        B = [b7(k,:,p)' b5(k,:,p)' b6(k,:,p)' b3(k,:,p)' b4(k,:,p)' b2(k,:,p)' b1(k,:,p)' b0(k,:,p)'];
        d = [0 1 nz1 nz1+1 nz1^2 nz1^2+1 nz1^2+nz1 nz1^2+nz1+1];
        At = spdiags(B,d,nu1, 4*nz1);
        [i, j, sn] = find(At);
        i = i + (k-1)*nu1 + (p-1)*nu1^2;
        j = j + (k-1)*nz1 + (p-1)*nz1^2;
        I(counter+1: counter+size(i,1)) = i;
        J(counter+1: counter+size(i,1)) = j;
        S(counter+1: counter+size(i,1)) = sn;
        counter = counter + size(i,1);
    end
end

Az = sparse(I(1:counter),J(1:counter),S(1:counter), m, nz);
Az(:, ~any(Az,1)) = [];
A = [Au Az];
B = A';
nnz(A)
size(B)
disp('aspect ratio')
ar = size(B,1)/size(B,2)
%%
file = strcat('../mats/invpoi/3d/invpoi_3d_',num2str(s),'.mm');
mmwrite(file,B); 




