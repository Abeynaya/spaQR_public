
FILE[0 ]="../mats/advdiff/2d/advdiff_2_128_1_p1_1000.mm"
FILE[1 ]="../mats/advdiff/2d/advdiff_2_256_1_p1_1000.mm"
# FILE[2 ]="../mats/advdiff/2d/advdiff_2_512_1_p1_1000.mm"
# FILE[3 ]="../mats/advdiff/2d/advdiff_2_1024_1_p1_1000.mm"

N=(128 256 512 1024)
T=(1e-1 1e-4)

for i in {0..3}
do
n=$((($i)/2))
t=$((($i)%2))
# srun ../spaQR matrix mesh_size dimension tolerance skip scale solver 
.././spaQR -m ${FILE[n]} -n ${N[n]} -d 2  -t ${T[t]}  --skip 4 --scale 1 --solver "GMRES"
.././spaQR -m ${FILE[n]} -n ${N[n]} -d 2  -t ${T[t]}  --skip 4 --scale 0 --solver "GMRES"
done

