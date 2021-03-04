
FILE[0 ]="../mats/invpoi/2d/invpoi_2d_2_128.mm"
FILE[1 ]="../mats/invpoi/2d/invpoi_2d_2_256.mm"

# Generate these matrices using code in ../matrix_gen
# FILE[2 ]="../mats/invpoi/2d/invpoi_2d_2_512.mm" 
# FILE[3 ]="../mats/invpoi/2d/invpoi_2d_2_1024.mm" 

N=(128 256 512 1024)
T=(1e-2 1e-4)

for i in {0..3}
do
n=$((($i)/2))
t=$((($i)%2))
# srun ../spaQR matrix mesh_size dimension tolerance skip 
.././spaQR -m ${FILE[n]} -n ${N[n]} -d 2  -t ${T[t]}  --skip 4 

done

