#!/bin/bash
nist_path='/gpfs/data/tserre/data/NIST/'
script_name='emnist_helpers.py'
username='jk9'

# submit job
PARTITION='batch' # batch # bibs-smp # bibs-gpu # gpu # small-batch
QOS='bibs-tserre-condo' # pri-jk9

for i_machine in $(seq 1 $n_machines); do
sbatch -J "C-NIST-$script_name[$i_machine]" <<EOF
#!/bin/bash
#SBATCH -p $PARTITION
#SBATCH -n 4
#SBATCH -t 1:00:00
#SBATCH --mem=16G
#SBATCH --begin=now
#SBATCH --qos=$QOS
#SBATCH --output=/gpfs/scratch/$username/slurm/slurm-%j.out
#SBATCH --error=/gpfs/scratch/$username/slurm/slurm-%j.out

echo "Starting job $i_machine on $HOSTNAME"
LC_ALL=en_US.utf8 \

python $script_name $nist_path
EOF
done
