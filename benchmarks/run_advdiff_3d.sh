
FILE[0 ]="../mats/advdiff/3d/advdiff_3_32_1_p1_1.mm"
FILE[1 ]="../mats/advdiff/3d/advdiff_3_64_1_p1_1.mm"
# FILE[2 ]="../mats/advdiff/3d/advdiff_3_80_1_p1_1.mm"
# FILE[3 ]="../mats/advdiff/3d/advdiff_3_96_1_p1_1.mm"

N=(32 64 80 96)
T=(1e-1 1e-2)

for i in {0..3}
do
n=$((($i)/2))
t=$((($i)%2))
# srun ../spaQR matrix mesh_size dimension tolerance skip scale solver 
.././spaQR -m ${FILE[n]} -n ${N[n]} -d 3  -t ${T[t]}  --skip 2 --scale 1 --solver "GMRES"
# .././spaQR -m ${FILE[n]} -n ${N[n]} -d 2  -t ${T[t]}  --skip 4 --scale 0 --solver "GMRES"
done

