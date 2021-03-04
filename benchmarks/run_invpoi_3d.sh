
FILE[0 ]="../mats/invpoi/3d/invpoi_3d_2_32.mm"
FILE[1 ]="../mats/invpoi/3d/invpoi_3d_2_48.mm"

# Generate these matrices using code in ../matrix_gen
# FILE[2 ]="../mats/invpoi/3d/invpoi_3d_2_64.mm" 
# FILE[3 ]="../mats/invpoi/3d/invpoi_3d_2_128.mm" 

N=(32 48 64 128)
T=(1e-1 1e-2)

for i in {0..3}
do
n=$((($i)/2))
t=$((($i)%2))
# srun ../spaQR matrix mesh_size dimension tolerance skip scale solver 
.././spaQR -m ${FILE[n]} -n ${N[n]} -d 3  -t ${T[t]}  --skip 2 
done

