#!/bin/bash
n_machines=50
num_images=5000
imgs_path='/gpfs/data/tserre/data/cluttered_nist3_locctrl_small/'
fonts_path='/users/jk9/projects/cluttered_nist/'
script_name='cluttered_nist_locctrl_wrapper.py'
username='jk9'

# submit job
PARTITION='batch' # batch # bibs-smp # bibs-gpu # gpu # small-batch
QOS='bibs-tserre-condo' # 'bibs-tserre-condo' # pri-jk9

for i_machine in $(seq 1 $n_machines); do
sbatch -J "C-NIST-$script_name[$i_machine]" <<EOF
#!/bin/bash
#SBATCH -p $PARTITION
#SBATCH -n 2
#SBATCH -t 4:00:00
#SBATCH --mem=8G
#SBATCH --begin=now
#SBATCH --account=$QOS
#SBATCH --output=/gpfs/scratch/$username/slurm/cnist3locc-$i_machine.out
#SBATCH --error=/gpfs/scratch/$username/slurm/cnist3locc-$i_machine.out

echo "Starting job $i_machine on $HOSTNAME"
LC_ALL=en_US.utf8 \
module load boost ffmpeg/1.2

python $script_name $n_machines $i_machine $num_images $imgs_path $fonts_path
EOF
done
